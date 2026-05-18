import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm
from basicsr.utils.registry import ARCH_REGISTRY
from PIL import Image
from PyDiff.scripts.utils import pad_tensor

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3, stretch=False):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
        if stretch:
            from scripts.utils import stretch_linear
            betas = stretch_linear(betas)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) /
                n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


@ARCH_REGISTRY.register()
class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            image_size,
            channels=3,
            loss_type='l1',
            conditional=True,
            schedule_opt=None,
            color_fn=None,
            color_limit=None,
            resize_all=False,
            resize_res=-1,
            pyramid_list=[1]
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        self.color_fn = color_fn
        if schedule_opt is not None:
            pass
        if color_limit:
            self.color_limit = color_limit
        else:
            self.color_limit = -1
        self.resize_all = resize_all
        self.resize_res = resize_res
        self.pyramid_list = pyramid_list

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        # betas = make_beta_schedule(
        #     schedule=schedule_opt['schedule'],
        #     n_timestep=schedule_opt['n_timestep'],
        #     linear_start=0.00004,
        #     linear_end=0.01,
        #     stretch=schedule_opt.get('stretch', False))
        betas = (
                torch.linspace(0.00004 ** 0.5, 0.01 ** 0.5, 1000, dtype=torch.float64) ** 2
        )
        # print(betas)
        # make downsampling schedule
        assert schedule_opt['n_timestep'] % len(self.pyramid_list) == 0
        segment_length = schedule_opt['n_timestep'] // len(self.pyramid_list)
        downsampling_schedule = []
        for downsampling_scale in self.pyramid_list:
            downsampling_schedule += [downsampling_scale] * segment_length
        downsampling_schedule = np.array(downsampling_schedule)
        downsampling_schedule = torch.from_numpy(downsampling_schedule).to(device)
        self.register_buffer('downsampling_schedule', downsampling_schedule)
        timesteps, = betas.shape
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        k = np.zeros(timesteps + 1)
        a2 = np.zeros(timesteps)
        k[0] = 1  # 初始值不变
        decay_coef = 0.999  # 关键调整：从0.999改为0.9993（可在0.998~0.9995间微调）
        for t in range(0, timesteps):
            k[t + 1] = k[t] * decay_coef  # 使用新的衰减系数
            # 计算a2前先确保分子非负（避免a2为负）
            numerator = k[t + 1] ** 2 - betas[t]
            # numerator = max(numerator, 1e-6)  # 强制分子≥1e-6，防止a2为负或过小
            a2[t] = numerator / (k[t] ** 4)
            # 限制a2的最大值，避免数值溢出（关键：防止a2过大导致alphas异常）
            # a2[t] = min(a2[t], 10.0)  # a2最大不超过10，可根据实际情况调整
        coef_k = k[1:]
        alphas = np.sqrt(a2)
        alphas2 = a2
        alphas2_k2 = alphas2*coef_k**2
        alphas_cumprod = np.cumprod(alphas2_k2, axis=0)

        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        # alphas_cumprod = np.clip(alphas_cumprod, 1e-8, None)  # 最小值1e-8，防止分母为0
        # alphas_cumprod_prev = np.clip(alphas_cumprod_prev, 1e-8, None)
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))
        self.register_buffer('coef_k', to_torch(coef_k))
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(coef_k**4 - alphas_cumprod)))
        # self.register_buffer('log_one_minus_alphas_cumprod',
        #                      to_torch(np.log(coef_k**4 - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(coef_k**4 / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
                             (k[0:-1]**4 - alphas_cumprod_prev) / (coef_k**2 - alphas_cumprod_prev*alphas2)
        # posterior_variance = np.clip(posterior_variance, 1e-8, None)  # 方差最小值1e-8
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        # self.register_buffer('posterior_log_variance_clipped', to_torch(
        #     np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (coef_k**2 - alphas_cumprod_prev*alphas2)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (k[0:-1]**4 - alphas_cumprod_prev) * alphas / (coef_k*(coef_k**2. - alphas_cumprod_prev*alphas2))))

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    # use ddim to sample
    @torch.no_grad()
    def ddim_pyramid_sample(
            self,
            x_in,
            pyramid_list,
            ddim_timesteps=50,
            ddim_discr_method="uniform",
            ddim_eta=0.0,
            clip_denoised=True,
            continous=False,
            return_x_recon=False,
            return_pred_noise=False,
            return_all=False,
            pred_type='noise',
            clip_noise=False,
            save_noise=False,
            color_gamma=None,
            color_times=1,
            fine_diffV2=False,
            fine_diffV2_st=200,
            fine_diffV2_num_timesteps=20,
            do_some_global_deg=False,
            use_up_v2=False):

        assert len(
            pyramid_list) == ddim_timesteps, f'len(pyramid_list):{len(pyramid_list)} != ddim_timesteps{ddim_timesteps}'

        if return_all:
            assert not (
                        return_x_recon or return_pred_noise), "[return_x_recon, return_pred_noise, return_all], choose one or not!"
        assert not (
                    return_x_recon and return_pred_noise), "[return_x_recon, return_pred_noise, return_all], choose one or not!"
        # make ddim timestep sequence
        if ddim_discr_method == 'uniform':
            c = self.num_timesteps // ddim_timesteps
            ddim_timestep_seq = list(reversed(range(self.num_timesteps - 1, -1, -c)))
            ddim_timestep_seq = np.asarray(ddim_timestep_seq)
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                    (np.linspace(0, np.sqrt(self.num_timesteps * .8), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([-1]), ddim_timestep_seq[:-1])

        device = x_in.device
        b, c, h, w = x_in[:, :3, :, :].shape
        init_h = h // pyramid_list[-1]
        init_w = w // pyramid_list[-1]
        # start from pure noise (for each example in the batch)
        sample_img = torch.randn((b, c, init_h, init_w), device=device)

        # xlow = x_in[:, :3, :, :]
        sample_inter = (1 | (ddim_timesteps // 10))
        ret_img = x_in[:, :3, :, :]
        # os.makedirs('sample_images', exist_ok=True)
        # sample_img = torch.clamp(sample_img, 0, 1)
        # img_np = (sample_img.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        # Image.fromarray(img_np).save(f'samples_noise.png')
        for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
            if return_all and i % sample_inter == 0:
                all_process = [F.interpolate(sample_img, (h, w))]
            t = torch.full((b,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(b, 1).to(x_in.device)
            # noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(b, 1).to('cpu')
            prefix = f'sample_img_{t.item()}'
            prev_t = torch.full((b,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)

            # get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, sample_img.shape)
            posterior_variance_t = self._extract(self.posterior_variance, t, sample_img.shape)
            coef_k_t = self._extract(self.coef_k, t, sample_img.shape)
            if i == 0:
                alpha_cumprod_t_prev = torch.ones_like(alpha_cumprod_t)
                coef_k_t_prev = torch.ones_like(coef_k_t)
            else:
                alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_t, sample_img.shape)
                coef_k_t_prev = self._extract(self.coef_k, prev_t, sample_img.shape)

            if pred_type == 'noise':
                # 2. predict noise using model
                pred_noise = self.denoise_fn(torch.cat([F.interpolate(x_in, sample_img.shape[2:]), sample_img], dim=1),
                                             noise_level)
                if clip_noise:
                    pred_noise = torch.clamp(pred_noise, -1, 1)

                # 3. get the predicted x_0
                pred_x0 = (sample_img - torch.sqrt((coef_k_t**4. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            else:
                assert False, "only pred noise"
            if return_all and i % sample_inter == 0:
                all_process.append(F.interpolate(pred_x0, (h, w)))

            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=0., max=1.)

            sample_already = False

            if i != 0 and pyramid_list[i] != pyramid_list[i - 1]:
                upsample_scale = pyramid_list[i] // pyramid_list[i - 1]
                pred_x0 = F.interpolate(pred_x0, scale_factor=upsample_scale)
                pred_noise = F.interpolate(pred_noise, scale_factor=upsample_scale)
                if use_up_v2:
                    noise = torch.randn_like(pred_x0)
                    sample_img = self.sqrt_alphas_cumprod[prev_t] * pred_x0 + \
                                 self.sqrt_one_minus_alphas_cumprod[prev_t] * noise

                    sample_already = True
            # save_image(torch.clamp(pred_x0, 0, 1), f"dataset/146/pred_x0/step_{i:04d}.png")
            # save_image(torch.clamp(pred_noise, 0, 1), f"dataset/146/pred_noise/step_{i:04d}.png")
            if not sample_already:
                # compute variance: "sigma_t(η)" -> see DDIM formula (16)
                # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
                sigmas_t = ddim_eta * torch.sqrt(posterior_variance_t)
                # sigmas_t = ddim_eta * torch.sqrt(
                #     (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))

                # compute "direction pointing to x_t" of DDIM formula (12)
                pred_dir_xt = torch.sqrt(coef_k_t_prev**4 - alpha_cumprod_t_prev - sigmas_t ** 2) * pred_noise

                # compute x_{t-1} of DDIM formula (12). Here pred_x0 means 1/(cumprod(k_t*a_t)*(x_t-sqrt(k_t^4-cumprod(k_t^2*a_t^2)*pred_noise)
                x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(pred_x0)
                # save sample images

                sample_img = x_prev


            if return_all and i % sample_inter == 0:
                all_process.append(F.interpolate(sample_img, (h, w)))
            # save_image(torch.clamp(sample_img, 0, 1), f"dataset/146/pred_x0/step_{i:04d}.png")
            # save_image(torch.clamp(pred_noise, 0, 1), f"dataset/146/pred_noise/step_{i:04d}.png")
            if i % sample_inter == 0:
                if return_x_recon:
                    ret_img = torch.cat([ret_img, F.interpolate(pred_x0, (h, w))], dim=0)
                elif return_pred_noise:
                    ret_img = torch.cat([ret_img, pred_noise], dim=0)
                elif return_all:
                    ret_img = torch.cat([ret_img, torch.cat(all_process, dim=0)], dim=0)
                else:
                    ret_img = torch.cat([ret_img, F.interpolate(sample_img, (h, w))], dim=0)
        if continous:
            return ret_img
        else:
            return sample_img

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, coef_k, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (

                continuous_sqrt_alpha_cumprod * x_start +
                (coef_k**4 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * noise
        )

    def p_losses_cs_pyramid(self, x_HR, x_SR, noise=None, different_t_in_one_batch=False, clip_noise=False,
                            t_range=None, cs_on_shift=False,
                            cs_shift_range=None, t_border=1000, input_mode=None, crop_size=None, divide=None,
                            shift_x_recon_detach=False, shift_x_recon_detach_range=0.47, frozen_denoise=False,
                            down_uniform=False, down_hw_split=False, pad_after_crop=False):
        assert input_mode is not None, "must indicate input_mode, [crop, pad]!!"
        if not t_range:
            t_range = [1, self.num_timesteps]
        [b, c, h, w] = x_HR.shape
        if different_t_in_one_batch:
            t = torch.randint(0, self.num_timesteps, (b,)).long() + 1
            t = t.to(x_HR.device)
            continuous_sqrt_alpha_cumprod = self._extract(torch.from_numpy(self.sqrt_alphas_cumprod_prev), t,
                                                          self.x_start.shape)
            coef_k = self._extract(torch.from_numpy(self.coef_k), t,
                                   self.x_start.shape)
            continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)
            coef_k = coef_k.view(b, -1)
        else:
            t = np.random.randint(t_range[0], t_range[1] + 1)  # [1, 2000] [1, 2001)
            # print(t)
            continuous_sqrt_alpha_cumprod = torch.FloatTensor(
                np.random.uniform(
                    self.sqrt_alphas_cumprod_prev[t],
                    self.sqrt_alphas_cumprod_prev[t],
                    size=b
                )
            ).to(x_HR.device)

            low = self.coef_k[t-1].cpu().detach().numpy()
            high = self.coef_k[t-1].cpu().detach().numpy()

            coef_k = torch.FloatTensor(
                np.random.uniform(
                    low,
                    high,
                    size=b
                )
            ).to(x_HR.device)
            # continuous_sqrt_alpha_cumprod 是 sqrt(γ)
            continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
                b, -1)
            coef_k = coef_k.view(
                b, -1)

        r = self.downsampling_schedule[t - 1]
        # print(self.downsampling_schedule)
        x_HR = F.interpolate(x_HR, (x_HR.shape[2] // r, x_HR.shape[3] // r))
        x_SR = F.interpolate(x_SR, (x_SR.shape[2] // r, x_SR.shape[3] // r))

        _, _, H, W = x_HR.shape
        if input_mode == 'crop':
            if isinstance(crop_size, int):
                crop_size = [crop_size, crop_size]
            if H < crop_size[0]:
                crop_size[0] = H
            if W < crop_size[1]:
                crop_size[1] = W
            h = np.random.randint(0, H - crop_size[0] + 1)
            w = np.random.randint(0, W - crop_size[1] + 1)
            x_HR = x_HR[:, :, h: h + crop_size[0], w: w + crop_size[1]]
            x_SR = x_SR[:, :, h: h + crop_size[0], w: w + crop_size[1]]
        elif input_mode == 'pad':
            assert False, "wait to do!!"

        if pad_after_crop:
            x_HR, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(x_HR, 16)
            x_SR, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(x_SR, 16)

        x_start = x_HR
        [b, c, h, w] = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), coef_k=coef_k.view(-1, 1, 1, 1), noise=noise)
        # save_image((x_noisy * 2 - 1), "x_noisy.png")
        if frozen_denoise:
            with torch.no_grad():
                if not self.conditional:
                    model_output = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
                else:
                    model_output = self.denoise_fn(
                        torch.cat([x_SR, x_noisy], dim=1), continuous_sqrt_alpha_cumprod)
        else:
            if not self.conditional:
                model_output = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
            else:
                model_output = self.denoise_fn(
                    torch.cat([x_SR, x_noisy], dim=1), continuous_sqrt_alpha_cumprod)
        if clip_noise:
            model_output = torch.clamp(model_output, -1, 1)
        self.noise = noise
        self.pred_noise = model_output
        self.pred_noise_detach = self.pred_noise.detach()
        self.x_start = x_start
        self.x_noisy = x_noisy

        self.x_recon = self.sqrt_recip_alphas_cumprod[t - 1] * x_noisy - self.sqrt_recipm1_alphas_cumprod[
            t - 1] * model_output
        # self.x_recon = torch.clamp(self.x_recon, 0, 1)
        self.t = t - 1

        self.x_recon_detach = self.x_recon.detach()

        self.x_recon_cs = self.x_recon_detach

        self.pred_noise_cs = (x_noisy - self.sqrt_alphas_cumprod[t - 1] * self.x_recon_cs) / \
                             self.sqrt_one_minus_alphas_cumprod[t - 1]
        # save_image((self.pred_noise_cs * 2 - 1), "Pred_noisy.png")
        continuous_sqrt_alpha_cumprod_mean = continuous_sqrt_alpha_cumprod.mean()
        continuous_alpha_cumprod = continuous_sqrt_alpha_cumprod_mean * continuous_sqrt_alpha_cumprod_mean
        color_scale = torch.sqrt((coef_k**4 - continuous_alpha_cumprod**2)) / continuous_alpha_cumprod
        return self.pred_noise, self.noise, self.x_recon_cs, self.x_start, self.t, color_scale



    def forward(self, x_HR, x_SR, train_type='ddpm', *args, **kwargs):
        kwargs_cp = kwargs.copy()
        for k in kwargs_cp:
            if kwargs[k] is None:
                kwargs.pop(k)
        if train_type == 'ddpm_cs_pyramid':
            return self.p_losses_cs_pyramid(x_HR, x_SR, *args, **kwargs)
        else:
            assert False, f"Wrong train_type={train_type}"
