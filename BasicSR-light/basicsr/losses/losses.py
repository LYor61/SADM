import math
import torch
from kornia.color import rgb_to_lab
from pytorch_msssim import MS_SSIM
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
from torchvision.transforms import transforms

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)

# @LOSS_REGISTRY.register()
# class GTmeanLoss(nn.Module):
#     """GTmean loss.
#
#     Args:
#         loss_weight (float): Loss weight for GTmean loss. Default: 1.0.
#         reduction (str): Specifies the reduction to apply to the output.
#             Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
#     """
#
#     def __init__(self, loss_weight='sigmoid', reduction='mean', sigma_=0.1):
#         super(GTmeanLoss, self).__init__()
#         if reduction not in ['none', 'mean', 'sum']:
#             raise ValueError(f'Unsupported reduction mode: {reduction}. '
#                              f'Supported ones are: {_reduction_modes}')
#
#         self.sigma = sigma_
#         print(f"current sigma is {self.sigma}")
#         if loss_weight == 'dou_KL_2':
#             self.iter_weight = self.double_KL_div_2
#         else:
#             assert False, 'Currently weight is undefined'
#
#         self.reduction = reduction
#         self.transform = transforms.Grayscale(num_output_channels=1)
#
#     def linear(self, num_iter):
#         return (1.0 * torch.tensor(num_iter) / 150000)
#
#     def KL_div(self, mu_1, mu_2, sigma_1, sigma_2):
#         return torch.log(sigma_2 / sigma_1) + 0.5 * (sigma_1 ** 2 + (mu_1 - mu_2) ** 2) / sigma_2 ** 2 - 0.5
#
#     def double_KL_div_2(self, mu_1, mu_2, sigma):
#         sigma_1 = sigma * mu_1
#         sigma_2 = sigma * mu_2
#         mu_M = 0.5 * (mu_1 + mu_2)
#         sigma_M = torch.sqrt(((sigma_1) ** 2 + (sigma_2) ** 2) / 2)
#         KL_2_weight = 0.5 * self.KL_div(mu_1, mu_M, sigma_1, sigma_M) + 0.5 * self.KL_div(mu_2, mu_M, sigma_2, sigma_M)
#
#         return KL_2_weight
#
#     def forward(self, pred, target, num_iter, weight=None, **kwargs):
#         """
#         Args:
#             pred (Tensor): of shape (N, C, H, W). Predicted tensor.
#             target (Tensor): of shape (N, C, H, W). Ground truth tensor.
#             weight (Tensor, optional): of shape (N, C, H, W). Element-wise
#                 weights. Default: None.
#         """
#         E_y = torch.mean(self.transform(target), dim=(1, 2, 3))
#         E_x = torch.mean(self.transform(pred), dim=(1, 2, 3))
#         # print(f"E_y contains NaN? {torch.isnan(E_y).any().item()}")
#         # print(f"E_x contains NaN? {torch.isnan(E_x).any().item()}")
#         self.loss_weight_unclip = self.iter_weight(torch.abs(E_y), torch.abs(E_x), self.sigma)
#         self.loss_weight = torch.clip(self.loss_weight_unclip, 0, 1).detach()
#         epsilon = 1e-8
#         m = E_y / (E_x+epsilon)
#         # print(f"m contains NaN? {torch.isnan(m).any().item()}")
#         pred_clip = torch.clip(m[:, None, None, None] * pred, 0, 1)
#
#         L1_loss = l1_loss(pred, target, self.loss_weight[:, None, None, None], reduction=self.reduction)
#         GT_loss = l1_loss(pred_clip, target, (1 - self.loss_weight)[:, None, None, None], reduction=self.reduction)
#         # print(f"pred_clip contains NaN? {torch.isnan(pred_clip).any().item()}")
#         # print(f"target contains NaN? {torch.isnan(target).any().item()}")
#         total_loss = GT_loss + L1_loss
#
#         return total_loss.mean()


@LOSS_REGISTRY.register()
class GTmeanLoss(nn.Module):
    """GTmean loss (增强数值稳定性版)."""

    def __init__(self, loss_weight='sigmoid', reduction='mean', sigma_=0.1):
        super(GTmeanLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.sigma = sigma_
        print(f"current sigma is {self.sigma}")
        if loss_weight == 'dou_KL_2':
            self.iter_weight = self.double_KL_div_2
        else:
            assert False, 'Currently weight is undefined'

        self.reduction = reduction
        self.transform = transforms.Grayscale(num_output_channels=1)
        # --- 新增：全局微小偏移，用于防止所有除以零和log(0)的场景 ---
        self.epsilon = 1e-8  # 经验值，可根据需求调整（1e-8 ~ 1e-6 均安全）
        self.loss_weight = None

    def linear(self, num_iter):
        return (1.0 * torch.tensor(num_iter, device=self.device) / 150000)  # --- 修复：确保tensor在正确设备上 ---

    def KL_div(self, mu_1, mu_2, sigma_1, sigma_2):
        # --- 关键修改：为sigma_1和sigma_2加偏移，防止log(0)和除以零 ---
        sigma_1_safe = sigma_1 + self.epsilon
        sigma_2_safe = sigma_2 + self.epsilon

        # 计算log项：使用torch.log1p避免log(0)，同时确保输入为正
        log_term = torch.log(sigma_2_safe / sigma_1_safe)
        # 计算平方项：确保分母不为零
        square_term = (sigma_1_safe ** 2 + (mu_1 - mu_2) ** 2) / (sigma_2_safe ** 2)

        kl = log_term + 0.5 * square_term - 0.5
        # --- 新增：剪辑极端值，防止KL散度过大导致梯度爆炸 ---
        kl_clipped = torch.clamp(kl, min=-1e3, max=1e3)  # 限制KL值在[-1e3, 1e3]，避免极端值
        return kl_clipped

    def double_KL_div_2(self, mu_1, mu_2, sigma):
        # --- 新增：给 mu_1 和 mu_2 加下限剪辑，强制远离 0 ---
        # mu_min：根据 sigma 调整，确保 sigma*mu_min 不小于 1e-6（避免 sigma_1 极端小）
        mu_min = 1e-5 / sigma  # 若 sigma=0.1，mu_min=1e-4；若 sigma=0.01，mu_min=1e-3
        mu_1_safe = torch.clamp(mu_1, min=mu_min)  # 强制 mu_1 >= mu_min
        mu_2_safe = torch.clamp(mu_2, min=mu_min)  # 强制 mu_2 >= mu_min

        # 后续计算逻辑不变，但使用 mu_1_safe/mu_2_safe
        sigma_1 = sigma * mu_1_safe
        sigma_2 = sigma * mu_2_safe
        mu_M = 0.5 * (mu_1_safe + mu_2_safe)

        sigma_M_sq = ((sigma_1 ** 2 + sigma_2 ** 2) / 2) + self.epsilon
        sigma_M = torch.sqrt(sigma_M_sq)

        # 调用 KL_div 计算
        kl1 = self.KL_div(mu_1_safe, mu_M, sigma_1, sigma_M)
        kl2 = self.KL_div(mu_2_safe, mu_M, sigma_2, sigma_M)
        KL_2_weight = 0.5 * kl1 + 0.5 * kl2

        # --- 增强：更严格的剪辑，确保权重不仅在 [0,1]，且避免接近 1 导致梯度集中 ---
        KL_2_weight_clipped = torch.clamp(KL_2_weight, min=0.01, max=0.99)  # 避免权重=0或1，梯度突变
        return KL_2_weight_clipped

    def forward(self, pred, target, num_iter, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        # --- 步骤 1: 计算所有需要的中间变量 ---
        E_y = torch.mean(self.transform(target), dim=(1, 2, 3))
        E_x = torch.mean(self.transform(pred), dim=(1, 2, 3))
        E_x_safe = E_x + self.epsilon

        # --- 新增：打印关键值，监控异常 ---
        mu_1 = torch.abs(E_y)
        mu_2 = torch.abs(E_x_safe)
        # print(f"[GTmeanLoss] mu_1 (abs E_y): {mu_1.mean().item():.6f}, mu_2 (abs E_x_safe): {mu_2.mean().item():.6f}")

        self.loss_weight_unclip = self.iter_weight(mu_1, mu_2, self.sigma)
        # print(f"[GTmeanLoss] loss_weight_unclip: {self.loss_weight_unclip.mean().item():.4f}")
        self.loss_weight = torch.clip(self.loss_weight_unclip, 0, 1).detach()

        # --- 步骤 3: 计算 m 和 pred_clip ---
        m = E_y / E_x_safe
        m_clipped = torch.clamp(m, min=0.1, max=10.0)
        pred_clip = torch.clip(m_clipped[:, None, None, None] * pred, 0, 1)

        # --- 步骤 4: 现在可以安全地使用 self.loss_weight 了 ---
        L1_loss = l1_loss(pred, target, self.loss_weight[:, None, None, None], reduction=self.reduction)
        GT_loss = l1_loss(pred_clip, target, (1 - self.loss_weight)[:, None, None, None], reduction=self.reduction)

        total_loss = GT_loss + L1_loss
        total_loss_clipped = torch.clamp(total_loss, min=0.0, max=1e5)

        if self.reduction == 'mean':
            return total_loss_clipped.mean()
        elif self.reduction == 'sum':
            return total_loss_clipped.sum()
        else:  # 'none'
            return total_loss_clipped

@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')

@LOSS_REGISTRY.register()
class LabColorLoss(nn.Module):
    """Lab颜色损失：专门约束a/b色度通道，解决色偏问题"""

    def __init__(self, color_weight=0.5, luminance_weight=0.5, loss_weight=1.0, reduction='mean', eps=1e-8):
        super().__init__()
        self.color_weight = color_weight
        self.luminance_weight = luminance_weight
        self.loss_weight = loss_weight  # 添加对loss_weight的支持
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target):
        # 转换到Lab，计算Loss逻辑...
        pred_lab = rgb_to_lab(pred)
        target_lab = rgb_to_lab(target)

        pred_L = pred_lab[:, 0, :, :]
        pred_a = pred_lab[:, 1, :, :]
        pred_b = pred_lab[:, 2, :, :]
        target_L = target_lab[:, 0, :, :]
        target_a = target_lab[:, 1, :, :]
        target_b = target_lab[:, 2, :, :]

        loss_L = F.l1_loss(pred_L, target_L, reduction=self.reduction)
        loss_a = F.l1_loss(pred_a, target_a, reduction=self.reduction)
        loss_b = F.l1_loss(pred_b, target_b, reduction=self.reduction)

        color_loss = (loss_a + loss_b) * 0.5 * self.color_weight
        luminance_loss = loss_L * self.luminance_weight
        total_loss = (color_loss + luminance_loss) * self.loss_weight

        return total_loss

@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

        Args:
            loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super(WeightedTVLoss, self).forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super(WeightedTVLoss, self).forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt, weights=None):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k], weights) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


@LOSS_REGISTRY.register()
class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


@LOSS_REGISTRY.register()
class MultiScaleGANLoss(GANLoss):
    """
    MultiScaleGANLoss accepts a list of predictions
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(MultiScaleGANLoss, self).__init__(gan_type, real_label_val, fake_label_val, loss_weight)

    def forward(self, input, target_is_real, is_disc=False):
        """
        The input is a list of tensors, or a list of (a list of tensors)
        """
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    # Only compute GAN loss for the last layer
                    # in case of multiscale feature matching
                    pred_i = pred_i[-1]
                # Safe operation: 0-dim tensor calling self.mean() does nothing
                loss_tensor = super().forward(pred_i, target_is_real, is_disc).mean()
                loss += loss_tensor
            return loss / len(input)
        else:
            return super().forward(input, target_is_real, is_disc)


def r1_penalty(real_pred, real_img):
    """R1 regularization for discriminator. The core idea is to
        penalize the gradient on real data alone: when the
        generator distribution produces the true data distribution
        and the discriminator is equal to 0 on the data manifold, the
        gradient penalty ensures that the discriminator cannot create
        a non-zero gradient orthogonal to the data manifold without
        suffering a loss in the GAN game.

        Ref:
        Eq. 9 in Which training methods for GANs do actually converge.
        """
    grad_real = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
    grad = autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()


def gradient_penalty_loss(discriminator, real_data, fake_data, weight=None):
    """Calculate gradient penalty for wgan-gp.

    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        weight (Tensor): Weight tensor. Default: None.

    Returns:
        Tensor: A tensor for gradient penalty.
    """

    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

    # interpolate between real_data and fake_data
    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if weight is not None:
        gradients = gradients * weight

    gradients_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    if weight is not None:
        gradients_penalty /= torch.mean(weight)

    return gradients_penalty


@LOSS_REGISTRY.register()
class GANFeatLoss(nn.Module):
    """Define feature matching loss for gans

    Args:
        criterion (str): Support 'l1', 'l2', 'charbonnier'.
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, criterion='l1', loss_weight=1.0, reduction='mean'):
        super(GANFeatLoss, self).__init__()
        if criterion == 'l1':
            self.loss_op = L1Loss(loss_weight, reduction)
        elif criterion == 'l2':
            self.loss_op = MSELoss(loss_weight, reduction)
        elif criterion == 'charbonnier':
            self.loss_op = CharbonnierLoss(loss_weight, reduction)
        else:
            raise ValueError(f'Unsupported loss mode: {criterion}. Supported ones are: l1|l2|charbonnier')

        self.loss_weight = loss_weight

    def forward(self, pred_fake, pred_real):
        num_d = len(pred_fake)
        loss = 0
        for i in range(num_d):  # for each discriminator
            # last output is the final prediction, exclude it
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.loss_op(pred_fake[i][j], pred_real[i][j].detach())
                loss += unweighted_loss / num_d
        return loss * self.loss_weight

@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, channel=3, data_range=1.0, reduction='mean', **kwargs):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.data_range = data_range  # 图像数据范围（如[0,1]）
        self.reduction = reduction

        # 创建高斯窗口（模拟人眼视觉特性）
        self.gaussian_window = self._create_gaussian_window(window_size, channel)

    def _create_gaussian_window(self, window_size, channel):
        """创建高斯窗口用于SSIM计算"""
        gauss = torch.Tensor([math.exp(-(x - window_size//2)** 2 / float(2 * 1.0 **2)) for x in range(window_size)])
        gauss = gauss / gauss.sum()
        window = gauss.unsqueeze(1) * gauss.unsqueeze(0)  # 2D高斯核
        window = window.unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1)  # 扩展到通道维度
        return window.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def _ssim(self, pred, gt):
        """手动计算SSIM"""
        # 确保输入形状匹配
        assert pred.shape == gt.shape, "预测图像与目标图像形状不匹配"

        mu_x = F.conv2d(pred, self.gaussian_window, padding=self.window_size//2, groups=self.channel)  # 均值
        mu_y = F.conv2d(gt, self.gaussian_window, padding=self.window_size//2, groups=self.channel)

        sigma_x = F.conv2d(pred**2, self.gaussian_window, padding=self.window_size//2, groups=self.channel) - mu_x**2  # 方差
        sigma_y = F.conv2d(gt**2, self.gaussian_window, padding=self.window_size//2, groups=self.channel) - mu_y**2
        sigma_xy = F.conv2d(pred*gt, self.gaussian_window, padding=self.window_size//2, groups=self.channel) - mu_x*mu_y  # 协方差

        C1 = (0.01 * self.data_range)** 2  # 稳定性常数
        C2 = (0.03 * self.data_range)**2

        ssim_map = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))

        if self.reduction == 'mean':
            return ssim_map.mean()
        elif self.reduction == 'sum':
            return ssim_map.sum()
        else:
            return ssim_map

    def forward(self, pred, gt):
        ssim_value = self._ssim(pred, gt)
        return 1 - ssim_value  # SSIM损失 = 1 - SSIM值


@LOSS_REGISTRY.register()
class MS_SSIMLoss(nn.Module):
    # 去掉 levels 参数（旧库不支持），添加 win_size 参数（减小窗口大小）
    def __init__(self, data_range=1.0, channel=3, win_size=7):
        super().__init__()
        # 传递 win_size 给 MS_SSIM，减小窗口大小（默认7，原默认11）
        self.ms_ssim = MS_SSIM(
            data_range=data_range,
            size_average=True,
            channel=channel,
            win_size=win_size  # 关键：减小窗口大小，降低图像尺寸要求
        )

    def forward(self, pred, target):
        return 1 - self.ms_ssim(pred, target)

    @LOSS_REGISTRY.register()  # 关键：注册到Loss注册表
    class GradientLoss(nn.Module):
        """梯度损失（增强图像边缘和细节一致性）
        计算预测图和目标图的梯度差异，支持L1/L2损失
        参数：
            loss_type: 损失类型（'l1' 或 'l2'，默认'l1'）
            loss_weight: 损失权重（默认1.0，兼容配置文件）
            channel: 图像通道数（默认3，RGB图像）
        """

        def __init__(self, loss_type='l1', loss_weight=1.0, channel=3):
            super().__init__()
            self.loss_type = loss_type.lower()
            self.loss_weight = loss_weight
            self.channel = channel

            # 定义Sobel算子（计算x、y方向梯度）
            self.sobel_x = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]] * channel,
                                        dtype=torch.float32)
            self.sobel_y = torch.tensor([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]] * channel,
                                        dtype=torch.float32)

        def forward(self, pred, target):
            """前向传播：计算梯度损失"""
            # 确保输入形状一致
            assert pred.shape == target.shape, f"输入形状不匹配：pred={pred.shape}, target={target.shape}"

            # 将Sobel算子移到输入设备（GPU/CPU）
            sobel_x = self.sobel_x.to(pred.device)
            sobel_y = self.sobel_y.to(pred.device)

            # 计算x、y方向梯度（使用卷积模拟Sobel算子）
            pred_grad_x = F.conv2d(pred, sobel_x, padding=1, groups=self.channel)
            pred_grad_y = F.conv2d(pred, sobel_y, padding=1, groups=self.channel)
            target_grad_x = F.conv2d(target, sobel_x, padding=1, groups=self.channel)
            target_grad_y = F.conv2d(target, sobel_y, padding=1, groups=self.channel)

            # 计算梯度差异损失
            if self.loss_type == 'l2':
                loss_x = F.mse_loss(pred_grad_x, target_grad_x, reduction='mean')
                loss_y = F.mse_loss(pred_grad_y, target_grad_y, reduction='mean')
            else:  # 默认l1损失
                loss_x = F.l1_loss(pred_grad_x, target_grad_x, reduction='mean')
                loss_y = F.l1_loss(pred_grad_y, target_grad_y, reduction='mean')

            # 总梯度损失 = （x方向损失 + y方向损失）/ 2 * 权重
            total_loss = (loss_x + loss_y) / 2 * self.loss_weight
            return total_loss