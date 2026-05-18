README.md
Single-Stage Signal Attenuation Diffusion Model for Low-Light Image Enhancement and Denoising
Official PyTorch implementation of SADM (Signal Attenuation Diffusion Model), arXiv:2604.05727.
https://doi.org/10.48550/arXiv.2604.05727Paper Link: 

📌 Abstract
Diffusion models excel at image restoration via probabilistic modeling of forward noise addition and reverse denoising, and their ability to handle complex noise while preserving fine details makes them well-suited for Low-Light Image Enhancement (LLIE). Mainstream diffusion based LLIE methods either adopt a two-stage pipeline or an auxiliary correction network to refine U-Net outputs, which severs the intrinsic link between enhancement and denoising and leads to suboptimal performance owing to inconsistent optimization objectives.
To address these issues, we propose the Signal Attenuation Diffusion Model (SADM), a novel diffusion process that integrates the signal attenuation mechanism into the diffusion pipeline, enabling simultaneous brightness adjustment and noise suppression in a single stage. Specifically, the signal attenuation coefficient simulates the inherent signal attenuation of low-light degradation in the forward noise addition process, encoding the physical priors of low-light degradation to explicitly guide reverse denoising toward the concurrent optimization of brightness recovery and noise suppression.
Our method eliminates the need for extra correction modules or staged training relied on by existing diffusion-based LLIE methods. We validate that our design maintains consistency with Denoising Diffusion Implicit Models (DDIM) via multi-scale pyramid sampling, balancing interpretability, restoration quality, and computational efficiency.

📁 Project Structure
├── BasicSR-light/         # Simplified BasicSR environment
├── PyDiff/                # Core diffusion model code
│   ├── archs/             # Network architecture (UNet, DDPM)
│   ├── data/              # Dataloader for LOL dataset
│   ├── models/            # Training & inference pipeline
│   ├── options/           # Configuration files
│   └── scripts/           # Common tools
├── dataset/               # Low-light dataset folder
├── pretrained_models/     # Pre-trained checkpoints
├── test.py                # Inference code
├── train.py               # Training code
├── metrics.py             # Evaluation metrics
├── environment.yml        # Conda environment configuration

🔧 Environment Installation
1. Create conda environment

conda env create -f environment.yml
conda activate torch39

2. Install dependencies

cd PyDiff
pip install -e .
cd ../BasicSR-light
pip install -e .

📊 Dataset Preparation
We use the public LOL dataset for training and evaluation:
- LOL-v1: 485 training pairs / 15 testing pairs
- LOL-v2-real: Real low-light scenes
- LOL-v2-syn: Synthetic low-light scenes
Put datasets into dataset/ folder and modify the yaml configuration file.
🚀 Quick Test (Inference)

python test.py -opt PyDiff/options/infer_v1.yaml

The enhanced images will be saved in the visualization folder.
🏋️ Training
Modify training parameters in PyDiff/options/train_v1.yaml, then run:

python train.py -opt PyDiff/options/train_v1.yaml

📈 Evaluation Metrics
This code supports mainstream low-light image evaluation metrics:
- PSNR
- SSIM
- LPIPS
📎 Pre-trained Weights
Pre-trained models for LOLv1 are placed in pretrained_models/lolv1.
You can directly use them from Baidu(https://pan.baidu.com/s/1HUDTolLpL4AXNI6BUxfF8Q?pwd=SADM 提取码: SADM) for inference without retraining.
📄 Citation
If you use this code for your research, please cite our paper:

@article{liu2026single,
  title={Single-Stage Signal Attenuation Diffusion Model for Low-Light Image Enhancement and Denoising},
  author={Liu, Ying and Zhang, Junchao and Wu, Caiyun},
  journal={arXiv preprint arXiv:2604.05727},
  year={2026},
  doi={10.48550/arXiv.2604.05727}
}

🪪 License
This project is open-source for academic research only. Commercial use is prohibited.
🌟 Star
If this repository helps you, please give a Star ⭐, thank you very much!
