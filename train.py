from basicsr.utils.registry import LOSS_REGISTRY

conflict_losses = ['L1Loss', 'MSELoss','GANLoss', 'MultiScaleGANLoss', 'CharbonnierLoss','WeightedTVLoss', 'PerceptualLoss', 'MS_SSIMLoss', 'GTmeanLoss']
for loss_name in conflict_losses:
    if loss_name in LOSS_REGISTRY._obj_map:
        del LOSS_REGISTRY._obj_map[loss_name]
# ==============================================================
import os.path as osp
from basicsr import parse_options
from basicsr.train import train_pipeline
import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")

from PyDiff.pydiff import archs, data, models
########### python pydiff/train.py -opt options/infer.yaml
####python train.py -opt PyDiff/options/train_v1.yaml
# import archs
# import data
# import NoP

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
