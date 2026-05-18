import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def ssim_lol(img, img2, crop_border=0, input_order='HWC', convert_to=None, **kwargs):
    """
    Calculate SSIM for LOL dataset
    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
        convert_to (str): Whether to convert the images to other color NoP. If None, the images are not altered.

    Returns:
        float: SSIM result.
    """
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')

    # Convert to HWC if needed
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
        img2 = img2.transpose(1, 2, 0)

    # Crop border if needed
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # Convert to Y channel if needed
    if convert_to == 'y':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Ensure images are in uint8 format
    img = img.astype(np.uint8)
    img2 = img2.astype(np.uint8)

    # Calculate SSIM with appropriate window size
    min_dim = min(img.shape[:2])

    # 如果图像太小，使用较小的窗口大小
    if min_dim < 7:
        win_size = min_dim if min_dim % 2 == 1 else min_dim - 1  # 确保窗口大小是奇数
        if win_size < 3:  # 如果窗口大小小于3，直接返回1（完全相似）或0（完全不相似）
            if np.array_equal(img, img2):
                return 1.0
            else:
                return 0.0
    else:
        win_size = 7  # 默认窗口大小

    # 计算SSIM
    try:
        # 尝试使用channel_axis参数（新版本skimage）
        if len(img.shape) == 2:  # 灰度图像
            score = ssim(img, img2, win_size=win_size)
        else:  # 彩色图像
            try:
                score = ssim(img, img2, win_size=win_size, channel_axis=-1)
            except TypeError:
                # 如果channel_axis不支持，回退到multichannel
                score = ssim(img, img2, win_size=win_size, multichannel=True)
    except ValueError as e:
        # 如果仍然出错，返回默认值
        print(f"SSIM calculation failed: {e}")
        if np.array_equal(img, img2):
            return 1.0
        else:
            return 0.0

    return score