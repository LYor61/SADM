import torch
from basicsr.utils.registry import METRIC_REGISTRY
from skimage.metrics import structural_similarity as ssim
import cv2

@METRIC_REGISTRY.register()  # 关键！注册到 METRIC_REGISTRY
def calculate_ssim_lol(img, img2, gray_scale=True,**kwargs):
    '''
    References:
    https://github.com/wyf0912/LLFlow/blob/f5ad48719285be2bc945ebccf8ad2338cad887f6/code/Measure.py#L34
    '''
    score, diff = ssim(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY), full=True, multichannel=True)
    # multichannel: If True, treat the last dimension of the array as channels. Similarity calculations are done independently for each channel then averaged.
    return score