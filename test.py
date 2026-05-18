import basicsr
from basicsr.utils.registry import LOSS_REGISTRY

conflict_losses = ['L1Loss', 'MSELoss','GANLoss', 'MultiScaleGANLoss', 'CharbonnierLoss','WeightedTVLoss', 'PerceptualLoss', 'MS_SSIMLoss', 'GTmeanLoss']
for loss_name in conflict_losses:
    if loss_name in LOSS_REGISTRY._obj_map:
        del LOSS_REGISTRY._obj_map[loss_name]
# ==============================================================
import logging
import torch
from os import path as osp

# from ssim_lol import ssim_lol

from PyDiff.pydiff.data.lol_dataset import LOL_Dataset
# set PYTHONUTF8=1
# python test.py -opt PyDiff/options/infer_v1.yaml
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options

def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    # make_exp_dirs(opt)

    make_exp_dirs(opt)

    from basicsr.utils.misc import mkdir_and_rename
    import os

    results_root = opt['path']['results_root']
    if os.path.exists(results_root):

        print(f"Directory {results_root} already exists, using it directly.")
    else:

        os.makedirs(results_root, exist_ok=True)

    opt['path']['log'] = os.path.join(results_root, 'log')
    opt['path']['visualization'] = os.path.join(results_root, 'visualization')

    os.makedirs(opt['path']['log'], exist_ok=True)
    os.makedirs(opt['path']['visualization'], exist_ok=True)


    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    logger.info(f"Results will be saved to: {opt['path']['results_root']}")
    logger.info(f"Visualizations will be saved to: {opt['path']['visualization']}")

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])


    logger.info(f"Test completed. Check results in: {opt['path']['results_root']}")

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
