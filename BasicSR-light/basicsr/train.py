import torch
# 强制只使用第0号GPU
torch.cuda.set_device(0)
# 或禁用多GPU（如果模型用了DataParallel，强制用单卡）
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只让系统看到第0号GPU
import datetime
import logging
import math
import time
import torch
import os  # 用于创建文件夹
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options


def init_tb_loggers(opt):
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('使用wandb时需开启tensorboard')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger


def create_train_val_dataloader(opt, logger, resume_state=None):
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(train_set, dataset_opt,
                                            num_gpu=opt['num_gpu'], dist=opt['dist'],
                                            sampler=train_sampler, seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))

            if resume_state:
                remaining_iters = total_iters - resume_state['iter']
                if remaining_iters <= 0:
                    logger.warning('已完成设定的总迭代数，自动终止恢复训练')
                    return None, None, None, 0, 0
                remaining_epochs = math.ceil(remaining_iters / num_iter_per_epoch)
                logger.info(f'恢复训练后：剩余迭代数={remaining_iters}, 剩余epoch={remaining_epochs}')

            logger.info('训练统计信息:' +
                        f'\n\t训练图片数量: {len(train_set)}' +
                        f'\n\t数据集放大系数: {dataset_enlarge_ratio}' +
                        f'\n\t单GPU批次大小: {dataset_opt["batch_size_per_gpu"]}' +
                        f'\n\tGPU数量: {opt["world_size"]}' +
                        f'\n\t每epoch迭代数: {num_iter_per_epoch}' +
                        f'\n\t总epoch: {total_epochs}; 总迭代数: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(val_set, dataset_opt,
                                          num_gpu=opt['num_gpu'], dist=opt['dist'],
                                          sampler=None, seed=opt['manual_seed'])
            logger.info(f'{dataset_opt["name"]} 验证集数量: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'不支持的数据集阶段: {phase}')
    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_dir = osp.join(opt['path']['experiments_root'], 'training_states')
        if osp.isdir(state_dir):
            state_files = list(scandir(state_dir, suffix='.state', recursive=False, full_path=False))
            if len(state_files) > 0:
                state_files.sort(key=lambda x: int(x.split('.state')[0]))
                resume_state_path = osp.join(state_dir, state_files[-1])
                opt['path']['resume_state'] = resume_state_path
                opt['resume_state'] = True
                logger = get_root_logger('basicsr', logging.INFO)
                logger.info(f'自动找到最新恢复文件: {resume_state_path}')
    elif opt['path'].get('resume_state') and osp.exists(opt['path']['resume_state']):
        resume_state_path = opt['path']['resume_state']
        opt['resume_state'] = True

    if resume_state_path is None:
        return None
    else:
        try:
            device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
            resume_state = torch.load(resume_state_path, map_location=device)
            check_resume(opt, resume_state['iter'])
            return resume_state
        except Exception as e:
            raise RuntimeError(f'加载恢复文件失败: {resume_state_path}\n错误信息: {str(e)}')


# ==============================
# 🔹 新增：Top3 模型保存逻辑
# ==============================
def update_best_models(model, epoch, current_iter, logger, save_dir):
    """维护并保存Top3模型 (基于 SSIM + PSNR/50)"""
    if not hasattr(model, 'best_models'):
        model.best_models = []

    eval_psnr = model.metric_results.get('psnr', 0)
    eval_ssim = model.metric_results.get('ssim', 0)
    score = eval_ssim + eval_psnr / 25

    save_path = osp.join(save_dir, f'epoch_{current_iter}.pth')

    # 🔹 修复：保存包含 'params' 键的字典
    state_dict = {
        'params': model.ddpm.state_dict(),
        'meta': {
            'iter': current_iter,
            'epoch': epoch,
            'ssim': eval_ssim,
            'psnr': eval_psnr,
            'score': score
        }
    }
    torch.save(state_dict, save_path)

    # 更新Top3列表
    model.best_models.append({'iter': current_iter, 'epoch': epoch,
                              'ssim': eval_ssim, 'psnr': eval_psnr,
                              'score': score, 'path': save_path})
    model.best_models = sorted(model.best_models, key=lambda x: x['score'], reverse=True)[:3]

    # 删除多余文件
    existing_files = [m['path'] for m in model.best_models]
    for f in os.listdir(save_dir):
        fp = osp.join(save_dir, f)
        if fp not in existing_files and f.endswith('.pth'):
            try:
                os.remove(fp)
            except:
                pass

    # 写入记录文件
    txt_path = osp.join(save_dir, 'best_models_info.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('Iter | Epoch | SSIM | PSNR | Score | Path\n')
        f.write('-' * 80 + '\n')
        for m in model.best_models:
            f.write(f"{m['iter']:>6} | {m['epoch']:>5} | {m['ssim']:.4f} | "
                    f"{m['psnr']:.2f} | {m['score']:.4f} | {m['path']}\n")

    logger.info(f"[Top3模型更新] 当前Iter {current_iter}: SSIM={eval_ssim:.4f}, "
                f"PSNR={eval_psnr:.2f}, Score={score:.4f} 已保存到 {save_path}")


# ==============================
# 🔹 新增：更新学习率工具函数（修复类型错误）
# ==============================
def safe_float_convert(value, param_name, logger):
    """安全地将值转换为float类型"""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError) as e:
        logger.error(f'❌ 配置参数 {param_name} 格式错误！请使用数值类型，当前值: {value} (类型: {type(value).__name__})')
        raise RuntimeError(f'参数 {param_name} 必须是数值类型（int/float），不能是字符串或其他类型') from e


def update_optimizer_lr(optimizer, new_lr, logger):
    """更新优化器的学习率（确保数值类型）"""
    if optimizer is None:
        return
    # 再次校验new_lr是数值类型
    if not isinstance(new_lr, (int, float)):
        new_lr = safe_float_convert(new_lr, 'new_lr', logger)

    for param_group in optimizer.param_groups:
        old_lr = param_group['lr']
        # 确保old_lr是数值类型
        if not isinstance(old_lr, (int, float)):
            old_lr = float(old_lr)

        param_group['lr'] = new_lr
        logger.info(f'📌 学习率已更新：{old_lr:.6f} → {new_lr:.6f}')


def train_pipeline(root_path):
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path
    torch.backends.cudnn.benchmark = True

    resume_custom_lr = safe_float_convert(opt['train'].get('resume_custom_lr'), 'resume_custom_lr', None)
    resume_lr_scale = safe_float_convert(opt['train'].get('resume_lr_scale'), 'resume_lr_scale', None)

    # 额外校验：resume_lr_scale不能是0或负数
    if resume_lr_scale is not None and resume_lr_scale <= 0:
        raise ValueError(f'❌ resume_lr_scale 必须是正数！当前值: {resume_lr_scale}')

    resume_state = load_resume_state(opt)
    opt['resume_state'] = resume_state is not None

    if resume_state is None:
        make_exp_dirs(opt)
        os.makedirs(osp.join(opt['path']['experiments_root'], 'training_states'), exist_ok=True)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))
    else:
        os.makedirs(osp.join(opt['path']['experiments_root'], 'training_states'), exist_ok=True)

    if resume_state is None:
        copy_opt_file(args.opt, opt['path']['experiments_root'])

    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    tb_logger = init_tb_loggers(opt)

    # 重新校验配置（此时logger已初始化，可以输出错误日志）
    resume_custom_lr = safe_float_convert(opt['train'].get('resume_custom_lr'), 'resume_custom_lr', logger)
    resume_lr_scale = safe_float_convert(opt['train'].get('resume_lr_scale'), 'resume_lr_scale', logger)
    if resume_lr_scale is not None and resume_lr_scale <= 0:
        logger.error(f'❌ resume_lr_scale 必须是正数！当前值: {resume_lr_scale}')
        raise ValueError(f'resume_lr_scale 必须是正数')

    result = create_train_val_dataloader(opt, logger, resume_state)
    if result is None:
        return
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    model = build_model(opt)
    if resume_state:
        model.ddpm.load_state_dict(resume_state['model'])
        if 'optimizer' in resume_state and resume_state['optimizer'] is not None:
            model.optimizer_g.load_state_dict(resume_state['optimizer'])
            # ==============================
            # 🔹 恢复后更新学习率（确保数值类型）
            # ==============================
            if resume_custom_lr is not None:
                # 使用固定的自定义学习率
                update_optimizer_lr(model.optimizer_g, resume_custom_lr, logger)
            elif resume_lr_scale is not None:
                # 使用缩放比例更新学习率
                current_lr = model.optimizer_g.param_groups[0]['lr']
                # 确保current_lr是数值类型
                current_lr = safe_float_convert(current_lr, 'current_lr', logger)
                new_lr = current_lr * resume_lr_scale
                update_optimizer_lr(model.optimizer_g, new_lr, logger)
            else:
                # 不修改，使用原学习率
                current_lr = model.optimizer_g.param_groups[0]['lr']
                current_lr = safe_float_convert(current_lr, 'current_lr', logger)
                logger.info(f'🔍 恢复默认学习率：{current_lr:.6f}')

        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
        logger.info(f'成功恢复训练：从epoch {start_epoch}, iter {current_iter} 开始')
    else:
        start_epoch = 0
        current_iter = 0

    msg_logger = MessageLogger(opt, current_iter, tb_logger)
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode', 'cpu')
    if prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        if not opt['datasets']['train'].get('pin_memory', False):
            raise ValueError('CUDAPrefetcher需开启pin_memory=True')
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'使用 {prefetch_mode} 预加载数据')
    else:
        raise ValueError(f'不支持的预加载模式 "{prefetch_mode}"')

    logger.info(f'训练启动：从epoch {start_epoch}, iter {current_iter} 开始')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    save_dir = osp.join(opt['path']['experiments_root'], 'models')
    os.makedirs(save_dir, exist_ok=True)  # 确保保存目录存在

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()
            if current_iter >= total_iters:
                break
            current_iter += 1
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            iter_timer.record()

            if current_iter == 1 and not opt['resume_state']:
                msg_logger.reset_start_time()

            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})  # 日志中会显示当前学习率
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info(f'保存模型和训练状态（iter: {current_iter}）')
                save_state = {
                    'epoch': epoch,
                    'iter': current_iter,
                    'opt': opt,
                    'model': model.ddpm.state_dict(),
                    'optimizer': model.optimizer_g.state_dict(),
                    'scheduler': None,
                    'data_pos': 0
                }
                state_path = osp.join(opt['path']['experiments_root'], 'training_states', f'{current_iter}.state')
                torch.save(save_state, state_path)
                logger.info(f'训练状态已保存到: {state_path}')
                model.save(epoch, current_iter)

            # 🔹 验证 + Top3 保存
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
                update_best_models(model, epoch, current_iter, logger, save_dir)

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'训练结束，总耗时: {consumed_time}')
    logger.info('保存最终模型和训练状态')
    final_model_path = osp.join(save_dir, 'final_model.pth')
    final_state_dict = {
        'params': model.ddpm.state_dict(),
        'meta': {
            'iter': total_iters,
            'epoch': total_epochs,
            'final': True
        }
    }
    torch.save(final_state_dict, final_model_path)
    logger.info(f'最终模型已保存到: {final_model_path}')
    model.save(epoch=-1, current_iter=-1)

    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
        update_best_models(model, total_epochs, total_iters, logger, save_dir)

    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)