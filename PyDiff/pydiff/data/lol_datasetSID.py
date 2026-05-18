import glob
import random
import os

import cv2
import math
import numpy as np
import torch
import torch.utils.data as data
from basicsr.utils.registry import DATASET_REGISTRY
from torchvision.transforms.functional import normalize
from PyDiff.scripts.utils import pad_tensor, hiseq_color_cv2_img, generate_position_encoding
from pydiff.data.haze import my_enhance_cuda


@DATASET_REGISTRY.register()
class LOL_Dataset(data.Dataset):
    def __init__(self, opt):
        super(LOL_Dataset, self).__init__()
        self.is_train = opt['name'] == 'train'
        self.opt = opt
        self.gt_root = opt['gt_root']
        self.input_root = opt['input_root']

        # 添加SID数据集支持
        self.dataset_type = opt.get('dataset_type', 'LOL')  # LOL, SID

        print(f"[{self.__class__.__name__}] - GT Root: {self.gt_root}")
        print(f"[{self.__class__.__name__}] - Input Root: {self.input_root}")
        print(f"[{self.__class__.__name__}] - Dataset Type: {self.dataset_type}")

        # SID数据集处理逻辑
        if self.dataset_type == 'SID':
            # SID: GT是long文件夹下的单张正常图像
            # Input是short文件夹下的多张低照度图像
            self.sid_data_pairs = self._load_sid_pairs()
            self.gt_paths = [pair['gt'] for pair in self.sid_data_pairs]
        else:
            # LOL: 常规的一对一映射
            self.gt_paths = glob.glob(os.path.join(self.gt_root, '*.png')) + \
                            glob.glob(os.path.join(self.gt_root, '*.jpg'))

        if len(self.gt_paths) == 0:
            raise ValueError(f"[{self.__class__.__name__}] No images found in GT root: {self.gt_root}")
        print(f"[{self.__class__.__name__}] Loaded {len(self.gt_paths)} GT images.")

        self.mean = self.opt.get('mean', [0, 0, 0])
        self.std = self.opt.get('std', [1, 1, 1])

        if self.dataset_type != 'SID':
            self.gt_paths.sort()

    def _load_sid_pairs(self):
        """加载SID数据集配对信息"""
        data_pairs = []

        # SID数据集结构：每个场景一个文件夹
        # GT: long文件夹下的每个场景文件夹中的一张正常曝光图像
        # LQ: short文件夹下的每个场景文件夹中的多张低照度图像

        # 获取所有场景文件夹
        scene_folders = sorted(os.listdir(self.gt_root))

        for scene_folder in scene_folders:
            gt_scene_path = os.path.join(self.gt_root, scene_folder)
            lq_scene_path = os.path.join(self.input_root, scene_folder)

            if not os.path.exists(lq_scene_path):
                print(f"Warning: LQ scene folder not found: {lq_scene_path}")
                continue

            # 获取GT图像（正常曝光，通常只有一张）
            gt_images = glob.glob(os.path.join(gt_scene_path, '*.png')) + \
                        glob.glob(os.path.join(gt_scene_path, '*.jpg')) + \
                        glob.glob(os.path.join(gt_scene_path, '*.tiff'))

            if not gt_images:
                print(f"Warning: No GT image found in {gt_scene_path}")
                continue

            gt_image = gt_images[0]  # SID中每个场景通常只有一张GT

            # 获取LQ图像（多张不同曝光）
            lq_images = sorted(glob.glob(os.path.join(lq_scene_path, '*.png')) +
                               glob.glob(os.path.join(lq_scene_path, '*.jpg')) +
                               glob.glob(os.path.join(lq_scene_path, '*.tiff')))

            # 为每个LQ图像创建一个配对
            for lq_image in lq_images:
                data_pairs.append({
                    'gt': gt_image,
                    'lq': lq_image,
                    'scene': scene_folder
                })

        print(
            f"[{self.__class__.__name__}] SID dataset: {len(data_pairs)} image pairs loaded from {len(scene_folders)} scenes")
        return data_pairs

    def __getitem__(self, index):
        if self.dataset_type == 'SID':
            # SID数据集处理
            pair = self.sid_data_pairs[index]
            gt_path = pair['gt']
            input_path = pair['lq']
            scene_name = pair['scene']
        else:
            # LOL数据集处理（保持原有逻辑）
            gt_path = self.gt_paths[index]
            gt_name = os.path.split(gt_path)[-1]
            input_path = os.path.join(self.input_root, gt_name)

        # 检查输入文件是否存在
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}\n"
                                    f"GT path: {gt_path}")

        # 读取图像
        gt_img = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB) / 255.0
        input_img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB) / 255.0

        # 后续处理保持不变...
        if self.opt.get('bright_aug', False):
            bright_aug_range = self.opt.get('bright_aug_range', [0.5, 1.5])
            input_img = input_img * np.random.uniform(*bright_aug_range)

        if self.opt.get('concat_with_hiseq', False):
            # dehaze = cv2.cvtColor(hiseq_color_cv2_img(cv2.imread(input_path)), cv2.COLOR_BGR2RGB) / 255.
            dehaze = my_enhance_cuda(input_path, train=self.is_train) / 255.0
            if self.opt.get('hiseq_random_cat', False) and np.random.uniform(0, 1) < self.opt.get('hiseq_random_cat_p',
                                                                                                  0.5):
                input_img = np.concatenate([dehaze, input_img], axis=2)
            else:
                input_img = np.concatenate([input_img, dehaze], axis=2)
            if self.opt.get('random_drop', False):
                if np.random.uniform() <= self.opt.get('random_drop_p', 1.0):
                    random_drop_val = self.opt.get('random_drop_val', 0)
                    if np.random.uniform() < 0.5:
                        input_img[:, :, :3] = random_drop_val
                    else:
                        input_img[:, :, 3:] = random_drop_val
            if self.opt.get('random_drop_hiseq', False):
                if np.random.uniform() < 0.5:
                    input_img[:, :, 3:] = 0

        # 水平翻转（如果启用）
        if self.opt.get('use_flip', False) and np.random.uniform() < 0.5:
            gt_img = cv2.flip(gt_img, 1, gt_img)
            input_img = cv2.flip(input_img, 1, input_img)

        # 生成低分辨率HQ图像（如果启用）
        if self.opt.get('input_with_low_resolution_hq', False):
            low_resolution_hq_size = self.opt.get('low_resolution_hq_size', 256)
            self.low_resolution_hq = cv2.resize(
                gt_img,
                (low_resolution_hq_size, low_resolution_hq_size)
            )

        # 拼接位置编码（如果启用）
        if self.opt.get('concat_with_position_encoding', False):
            H, W, _ = input_img.shape
            L = self.opt.get('position_encoding_L', 1)
            position_encoding = generate_position_encoding(H, W, L)
            input_img = np.concatenate([input_img, position_encoding], axis=2)

        # 调整大小（如果启用）
        if self.opt.get('resize', False):
            resize_size = self.opt['resize_size']
            if self.opt.get('resize_nearest', False):
                gt_img = cv2.resize(gt_img, dsize=(resize_size[1], resize_size[0]), interpolation=cv2.INTER_NEAREST)
                input_img = cv2.resize(input_img, dsize=(resize_size[1], resize_size[0]),
                                       interpolation=cv2.INTER_NEAREST)
            else:
                gt_img = cv2.resize(gt_img, dsize=(resize_size[1], resize_size[0]))
                input_img = cv2.resize(input_img, dsize=(resize_size[1], resize_size[0]))

        # 裁剪（如果启用）
        if self.opt['input_mode'] == 'crop':
            crop_size = self.opt['crop_size']
            H, W, _ = input_img.shape
            assert input_img.shape[:2] == gt_img.shape[:2], f"{input_img.shape}, {gt_img.shape}, {gt_path}"
            h = np.random.randint(0, H - crop_size + 1)
            w = np.random.randint(0, W - crop_size + 1)
            gt_img = gt_img[h: h + crop_size, w: w + crop_size, :]
            input_img = input_img[h: h + crop_size, w: w + crop_size, :]

        # 填充（如果启用）
        if self.opt['input_mode'] == 'pad':
            divide = self.opt['divide']
            gt_img_pt = torch.from_numpy(gt_img.transpose((2, 0, 1)))
            input_img_pt = torch.from_numpy(input_img.transpose((2, 0, 1)))
            gt_img_pt = torch.unsqueeze(gt_img_pt, 0)
            input_img_pt = torch.unsqueeze(input_img_pt, 0)
            gt_img_pt, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(gt_img_pt, divide)
            input_img_pt, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input_img_pt, divide)
            gt_img_pt = gt_img_pt[0, ...]
            input_img_pt = input_img_pt[0, ...]
            gt_img = gt_img_pt.numpy().transpose((1, 2, 0))
            input_img = input_img_pt.numpy().transpose((1, 2, 0))

        # 转换为tensor并归一化
        gt_img_pt = torch.from_numpy(gt_img.transpose((2, 0, 1))).float()
        input_img_pt = torch.from_numpy(input_img.transpose((2, 0, 1))).float()

        if hasattr(self, 'low_resolution_hq'):
            self.low_resolution_hq = torch.from_numpy(
                self.low_resolution_hq.transpose((2, 0, 1))
            ).float()

        # 归一化处理 0~1
        normalize(input_img_pt, [0.0] * input_img_pt.shape[0], [1.0] * input_img_pt.shape[0], inplace=True)
        normalize(gt_img_pt, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], inplace=True)
        if hasattr(self, 'low_resolution_hq'):
            # 对低分辨率HQ图像同样修改归一化
            normalize(
                self.low_resolution_hq,
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                inplace=True
            )

        # 构建返回字典
        return_dict = {"LR": input_img_pt, "HR": gt_img_pt,
                       "lq_path": input_path if self.dataset_type == 'SID' else gt_path}

        if self.dataset_type == 'SID':
            return_dict["scene_name"] = scene_name

        if self.opt['input_mode'] == 'pad':
            return_dict["pad_left"] = pad_left
            return_dict["pad_right"] = pad_right
            return_dict["pad_top"] = pad_top
            return_dict["pad_bottom"] = pad_bottom
        if self.opt.get('input_with_low_resolution_hq', False):
            return_dict["low_resolution_hq"] = self.low_resolution_hq

        return return_dict

    def __len__(self):
        if self.dataset_type == 'SID':
            return len(self.sid_data_pairs)
        else:
            return len(self.gt_paths)