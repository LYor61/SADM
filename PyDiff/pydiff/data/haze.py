import cv2
import torch
import numpy as np
import torch.nn.functional as F

def my_enhance_cuda(image_path, output_path=None, train=True):
    # 读取图像
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"无法读取图像，请检查路径: {image_path}")

    # 准备设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 转换为RGB张量（维度：1, 3, H, W）
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    epsilon = 1e-8  # 数值稳定性保护
    t0 = torch.mean(tensor)  # 原始图像均值
    w = 1.0 - t0  # ω = 1 - t0
    J = torch.amin(1.0 - tensor, dim=1, keepdim=True)  # 反转图像的通道最小值
    denominator = t0 + w * (1.0 - w * J)  # 分母
    # 防止除零 + 数值裁剪（避免超出[0,1]范围）
    img_out_tensor = tensor / torch.clamp(denominator, min=epsilon)
    img_out_tensor = torch.clamp(img_out_tensor, 0.0, 1.0)  # 保证像素值在合理范围

    # --- 使用 conv2d 进行颜色空间转换 ---

    # 1. RGB -> YCbCr
    rgb_to_ycbcr_matrix = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ], dtype=torch.float32, device=device).view(3, 3, 1, 1)

    ycbcr_offset = torch.tensor([0.0, 128.0 / 255.0, 128.0 / 255.0],
                                dtype=torch.float32, device=device).view(1, 3, 1, 1)

    ycbcr_tensor = F.conv2d(img_out_tensor, rgb_to_ycbcr_matrix, bias=None, stride=1, padding=0) + ycbcr_offset

    # gamaV 处理
    v_channel = ycbcr_tensor[:, 0, :, :]
    m = torch.mean(1.0 - tensor, dim=1) - 0.1
    v_channel_gamma = torch.pow(torch.clamp(v_channel, min=1e-8), m)
    ycbcr_tensor[:, 0, :, :] = v_channel_gamma

    # 2. YCbCr -> RGB
    ycbcr_to_rgb_matrix = torch.tensor([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ], dtype=torch.float32, device=device).view(3, 3, 1, 1)

    ycbcr_tensor_no_offset = ycbcr_tensor - ycbcr_offset
    final_rgb_tensor = F.conv2d(ycbcr_tensor_no_offset, ycbcr_to_rgb_matrix, bias=None, stride=1, padding=0)
    final_rgb_tensor = torch.clamp(final_rgb_tensor, 0.0, 1.0)  # ← “测试模式”要用的结果

    if train:
        used_tensor = final_rgb_tensor
    else:
        used_tensor = img_out_tensor

    # --- 准备输出 ---
    img_out_rgb_np = (used_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    if output_path:
        img_out_bgr = cv2.cvtColor(img_out_rgb_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, img_out_bgr)

    return img_out_rgb_np

if __name__ == "__main__":
    input_image_path = "../../../1.png"
    result = my_enhance_cuda(input_image_path, train=True)