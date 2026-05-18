import os
import cv2
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
import traceback

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


def read_image_safely(path: str) -> np.ndarray:
    """安全读取图像，支持中文路径"""
    try:
        with open(path, 'rb') as f:
            img_bytes = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if img is None:
            print(f"警告: OpenCV无法解码图像 {path}")
            return None

        # 确保是RGB格式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        print(f"无法读取图像 {path}: {str(e)}")
        return None


def calculate_metrics_opencv_skimage(gt_img: np.ndarray, pred_img: np.ndarray, crop_border: int = 0) -> Tuple[
    float, float]:
    """使用OpenCV和scikit-image计算PSNR和SSIM（输入：RGB格式）"""
    if crop_border > 0:
        gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border]
        pred_img = pred_img[crop_border:-crop_border, crop_border:-crop_border]

    if gt_img.shape != pred_img.shape:
        print(f"  调整预测图像尺寸: {pred_img.shape} -> {gt_img.shape}")
        pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))

    try:
        psnr_value = psnr(gt_img, pred_img, data_range=255)
        ssim_value = ssim(gt_img, pred_img, data_range=255, channel_axis=2,
                          win_size=min(7, min(gt_img.shape[0], gt_img.shape[1]) - 1))
        return psnr_value, ssim_value
    except Exception as e:
        print(f"  PSNR/SSIM计算失败: {str(e)}")
        return 0.0, 0.0


def calculate_lpips_official(gt_img: np.ndarray, pred_img: np.ndarray, model: lpips.LPIPS,
                             crop_border: int = 0) -> float:
    """使用官方LPIPS库计算LPIPS（输入：RGB格式）"""
    if crop_border > 0:
        gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border]
        pred_img = pred_img[crop_border:-crop_border, crop_border:-crop_border]

    if gt_img.shape != pred_img.shape:
        pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))

    def preprocess_image(img: np.ndarray) -> torch.Tensor:
        img_rgb = img.astype(np.float32) / 255.0
        img_rgb = (img_rgb * 2) - 1  # 归一化到[-1, 1]
        return torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    try:
        gt_tensor = preprocess_image(gt_img)
        pred_tensor = preprocess_image(pred_img)

        with torch.no_grad():
            lpips_value = model(gt_tensor, pred_tensor).item()

        return lpips_value
    except Exception as e:
        print(f"  LPIPS计算失败: {str(e)}")
        return 1.0  # 最差值


def get_image_pairs(gt_folder: str, pred_folder: str) -> List[Tuple[str, str, str]]:
    """获取匹配的图像对"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    gt_images = {}
    pred_images = {}

    # 扫描真值文件夹
    for root, _, files in os.walk(gt_folder):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext.lower() in valid_extensions:
                gt_images[name] = os.path.join(root, file)

    # 扫描预测文件夹
    for root, _, files in os.walk(pred_folder):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext.lower() in valid_extensions:
                pred_images[name] = os.path.join(root, file)

    common_files = set(gt_images.keys()) & set(pred_images.keys())
    pairs = [(filename, gt_images[filename], pred_images[filename]) for filename in sorted(common_files)]

    print(f"真值图像数: {len(gt_images)}, 预测图像数: {len(pred_images)}, 匹配对数: {len(pairs)}")
    return pairs


def calculate_fid_simple(gt_folder: str, pred_folder: str, crop_border: int = 0) -> float:
    """使用pytorch-fid库计算FID"""
    try:
        from pytorch_fid import fid_score
        import tempfile

        print("使用pytorch-fid计算FID...")

        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            gt_temp = os.path.join(temp_dir, "gt")
            pred_temp = os.path.join(temp_dir, "pred")
            os.makedirs(gt_temp, exist_ok=True)
            os.makedirs(pred_temp, exist_ok=True)

            # 复制图像到临时目录
            image_pairs = get_image_pairs(gt_folder, pred_folder)
            copied_count = 0

            for filename, gt_path, pred_path in tqdm(image_pairs, desc="准备FID图像"):
                gt_img = read_image_safely(gt_path)
                pred_img = read_image_safely(pred_path)

                if gt_img is not None:
                    if crop_border > 0:
                        gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border]
                    # 保存为BGR格式（OpenCV默认）
                    gt_img_bgr = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(gt_temp, f"{filename}.png"), gt_img_bgr)

                if pred_img is not None:
                    if crop_border > 0:
                        pred_img = pred_img[crop_border:-crop_border, crop_border:-crop_border]
                    pred_img_bgr = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(pred_temp, f"{filename}.png"), pred_img_bgr)
                    copied_count += 1

            print(f"成功准备 {copied_count} 对图像用于FID计算")

            if copied_count == 0:
                print("错误: 没有有效的图像对用于FID计算")
                return float('inf')

            # 计算FID
            fid_value = fid_score.calculate_fid_given_paths(
                [gt_temp, pred_temp],
                batch_size=1,
                device=device.type,
                dims=2048,
                num_workers=0
            )

            return fid_value

    except Exception as e:
        print(f"FID计算失败: {str(e)}")
        traceback.print_exc()
        return float('inf')


def evaluate_single_image_metrics(gt_folder: str, pred_folder: str, crop_border: int = 0) -> Tuple[
    List[dict], List[str]]:
    """评估单张图像指标：PSNR/SSIM/LPIPS(vgg/alex/squeeze)"""
    image_pairs = get_image_pairs(gt_folder, pred_folder)
    if not image_pairs:
        raise ValueError("未找到匹配的图像文件")

    print(f"找到 {len(image_pairs)} 对匹配图像，开始单张图像评估...")

    # 初始化三种LPIPS模型
    lpips_models = {}
    try:
        lpips_models['vgg'] = lpips.LPIPS(net='vgg').to(device)
        lpips_models['vgg'].eval()
        print("LPIPS(VGG)模型加载成功")
    except Exception as e:
        print(f"LPIPS(VGG)模型加载失败: {str(e)}")
        lpips_models['vgg'] = None

    try:
        lpips_models['alex'] = lpips.LPIPS(net='alex').to(device)
        lpips_models['alex'].eval()
        print("LPIPS(AlexNet)模型加载成功")
    except Exception as e:
        print(f"LPIPS(AlexNet)模型加载失败: {str(e)}")
        lpips_models['alex'] = None

    try:
        lpips_models['squeeze'] = lpips.LPIPS(net='squeeze').to(device)
        lpips_models['squeeze'].eval()
        print("LPIPS(SqueezeNet)模型加载成功")
    except Exception as e:
        print(f"LPIPS(SqueezeNet)模型加载失败: {str(e)}")
        lpips_models['squeeze'] = None

    results = []
    skip_log = []

    for filename, gt_path, pred_path in tqdm(image_pairs, desc="单张图像评估"):
        gt_img = read_image_safely(gt_path)
        pred_img = read_image_safely(pred_path)

        if gt_img is None or pred_img is None:
            skip_log.append(f"无法读取: {filename}")
            continue

        try:
            # 有参考指标
            psnr_val, ssim_val = calculate_metrics_opencv_skimage(gt_img, pred_img, crop_border)

            # 三种LPIPS计算
            lpips_vals = {}
            for net_name, model in lpips_models.items():
                if model is not None:
                    lpips_vals[net_name] = calculate_lpips_official(gt_img, pred_img, model, crop_border)
                else:
                    lpips_vals[net_name] = 1.0  # 加载失败时的默认值

            results.append({
                'filename': filename,
                'psnr': psnr_val,
                'ssim': ssim_val,
                'lpips_vgg': lpips_vals['vgg'],
                'lpips_alex': lpips_vals['alex'],
                'lpips_squeeze': lpips_vals['squeeze']
            })

            # 显示当前结果
            print(
                f"  {filename}: PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}, "
                f"LPIPS(VGG)={lpips_vals['vgg']:.4f}, LPIPS(Alex)={lpips_vals['alex']:.4f}, "
                f"LPIPS(Squeeze)={lpips_vals['squeeze']:.4f}"
            )

        except Exception as e:
            error_msg = f"计算失败: {filename} ({str(e)})"
            print(f"  {error_msg}")
            skip_log.append(error_msg)
            continue

    print(f"\n单张图像评估完成: 成功 {len(results)} 张, 跳过 {len(skip_log)} 张")
    return results, skip_log


def save_results_to_txt(results: List[dict], skip_log: List[str], fid_value: float,
                        gt_folder: str, pred_folder: str, output_path: str = None) -> str:
    """保存所有结果到TXT文件"""
    if output_path is None:
        output_path = os.path.join(pred_folder, "comprehensive_quality_metrics.txt")

    # 计算平均值
    if results:
        avg_psnr = np.mean([r['psnr'] for r in results])
        avg_ssim = np.mean([r['ssim'] for r in results])
        avg_lpips_vgg = np.mean([r['lpips_vgg'] for r in results])
        avg_lpips_alex = np.mean([r['lpips_alex'] for r in results])
        avg_lpips_squeeze = np.mean([r['lpips_squeeze'] for r in results])
    else:
        avg_psnr = avg_ssim = avg_lpips_vgg = avg_lpips_alex = avg_lpips_squeeze = 0.0

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("全面图像质量评估报告（含LPIPS多模型对比）\n")
        f.write("=" * 120 + "\n\n")

        f.write(f"真值文件夹: {gt_folder}\n")
        f.write(f"预测文件夹: {pred_folder}\n")
        f.write(f"评估时间: {np.datetime64('now')}\n")
        f.write(f"有效图像数: {len(results)} | 跳过数: {len(skip_log)}\n\n")

        # 数据集级指标
        f.write("【数据集级指标】\n")
        f.write("-" * 40 + "\n")
        f.write(f"FID (Fréchet Inception Distance): {fid_value:.4f}\n")
        f.write("  - 解释: 衡量两个数据集分布的相似度，值越小表示生成图像质量越好\n")
        f.write("  - 范围: 通常0-100，理想情况<50，优秀情况<20\n\n")

        # 单张平均指标
        f.write("【单张图像平均指标】\n")
        f.write("-" * 40 + "\n")
        f.write(f"平均PSNR (峰值信噪比): {avg_psnr:.4f} dB\n")
        f.write("  - 解释: 衡量图像像素级相似度，值越大越好\n")
        f.write("  - 范围: 通常20-40dB，>30dB表示质量较好\n\n")

        f.write(f"平均SSIM (结构相似性): {avg_ssim:.4f}\n")
        f.write("  - 解释: 衡量图像结构信息保持度，值越大越好\n")
        f.write("  - 范围: 0-1，>0.9表示质量优秀\n\n")

        f.write(f"平均LPIPS (VGG): {avg_lpips_vgg:.4f}\n")
        f.write("  - 解释: 基于VGG特征的感知相似度，值越小越好\n")
        f.write("  - 范围: 0-1，<0.3表示感知质量较好\n\n")

        f.write(f"平均LPIPS (AlexNet): {avg_lpips_alex:.4f}\n")
        f.write("  - 解释: 基于AlexNet特征的感知相似度，值越小越好\n")
        f.write("  - 范围: 0-1，<0.3表示感知质量较好\n\n")

        f.write(f"平均LPIPS (SqueezeNet): {avg_lpips_squeeze:.4f}\n")
        f.write("  - 解释: 基于SqueezeNet特征的感知相似度，值越小越好\n")
        f.write("  - 范围: 0-1，<0.3表示感知质量较好\n\n")

        # 详细结果
        f.write("【单张图像详细结果】\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'文件名':<25} {'PSNR(dB)':<10} {'SSIM':<10} {'LPIPS(VGG)':<12} {'LPIPS(Alex)':<12} {'LPIPS(Squeeze)':<12}\n")
        f.write("-" * 120 + "\n")
        for res in results:
            f.write(
                f"{res['filename']:<25} {res['psnr']:<10.4f} {res['ssim']:<10.4f} "
                f"{res['lpips_vgg']:<12.4f} {res['lpips_alex']:<12.4f} {res['lpips_squeeze']:<12.4f}\n"
            )

        # 跳过记录
        if skip_log:
            f.write("\n【跳过/失败记录】\n")
            f.write("-" * 50 + "\n")
            for log in skip_log:
                f.write(f"- {log}\n")

    print(f"详细结果已保存至: {output_path}")
    return output_path


def print_summary(results: List[dict], fid_value: float, output_file: str):
    """打印评估摘要"""
    if not results:
        print("无有效评估结果")
        return

    avg_psnr = np.mean([r['psnr'] for r in results])
    avg_ssim = np.mean([r['ssim'] for r in results])
    avg_lpips_vgg = np.mean([r['lpips_vgg'] for r in results])
    avg_lpips_alex = np.mean([r['lpips_alex'] for r in results])
    avg_lpips_squeeze = np.mean([r['lpips_squeeze'] for r in results])

    print("\n" + "=" * 80)
    print("图像质量评估完成！（含LPIPS多模型对比）")
    print("=" * 80)
    print(f"📊 有参考指标 (与真值比较):")
    print(f"   PSNR: {avg_psnr:.4f} dB (越高越好)")
    print(f"   SSIM: {avg_ssim:.4f} (越高越好)")
    print(f"\n🔍 LPIPS多模型对比 (越低越好):")
    print(f"   LPIPS(VGG): {avg_lpips_vgg:.4f}")
    print(f"   LPIPS(AlexNet): {avg_lpips_alex:.4f}")
    print(f"   LPIPS(SqueezeNet): {avg_lpips_squeeze:.4f}")
    print(f"\n🌐 数据集级指标:")
    print(f"   FID: {fid_value:.4f} (越低越好)")
    print(f"\n💾 结果文件: {output_file}")
    print("=" * 80)


def main():
    """主函数"""
    # 配置路径
    # GT_FOLDER = "dataset/sony/long"  # 真值图像文件夹
    # PRED_FOLDER = "D:/test_SONY/visualization"  # 预测图像文件夹
    GT_FOLDER = "D:/SecondPaper/results/lolv1/high"  # 真值图像文件夹
    PRED_FOLDER = 'D:/LLIE/Diffusion/results/lol_v1_hazeN1/visualization'  # 预测图像文件夹
    CROP_BORDER = 0  # 裁剪边界

    print("开始全面图像质量评估...")
    print(f"真值路径: {GT_FOLDER}")
    print(f"预测路径: {PRED_FOLDER}")
    print(f"裁剪边界: {CROP_BORDER}")

    try:
        # 步骤1: 检查图像格式
        print("\n步骤1: 检查图像格式...")
        image_pairs = get_image_pairs(GT_FOLDER, PRED_FOLDER)
        if not image_pairs:
            print("错误: 未找到匹配的图像文件")
            return

        # 步骤2: 评估单张图像指标
        print("\n步骤2: 评估单张图像指标（含LPIPS多模型）...")
        results, skip_log = evaluate_single_image_metrics(GT_FOLDER, PRED_FOLDER, CROP_BORDER)

        if not results:
            print("错误: 没有成功的单张图像评估结果")
            if skip_log:
                print("失败原因:")
                for log in skip_log:
                    print(f"  - {log}")
            return

        # 步骤3: 计算FID
        print("\n步骤3: 计算数据集级FID指标...")
        fid_val = calculate_fid_simple(GT_FOLDER, PRED_FOLDER, CROP_BORDER)

        # 步骤4: 保存结果
        print("\n步骤4: 保存评估结果...")
        output_file = save_results_to_txt(results, skip_log, fid_val, GT_FOLDER, PRED_FOLDER)

        # 步骤5: 打印摘要
        print_summary(results, fid_val, output_file)

    except Exception as e:
        print(f"\n❌ 评估过程出错: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()