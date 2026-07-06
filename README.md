# Single-Stage Signal Attenuation Diffusion Model for Low-Light Image Enhancement and Denoising
**Official PyTorch implementation of SADM (arXiv:2604.05727)**

[![arXiv](https://img.shields.io/badge/arXiv-2604.05727-b31b1b.svg)](https://arxiv.org/abs/2604.05727)
[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2604.05727-228be6)](https://doi.org/10.48550/arXiv.2604.05727)
[![Python](https://img.shields.io/badge/Python-3.9+-34d399.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-f97316.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic%20Only-red.svg)](#license)

---
---
## 🗂️ Project Structure
```
SADM/
├── BasicSR-light/         # Simplified BasicSR environment
├── PyDiff/                # Core diffusion code
│   ├── archs/             # UNet & DDPM architecture
│   ├── data/              # LOL dataset dataloader
│   ├── models/            # Train & inference pipeline
│   ├── options/           # YAML configuration files
│   └── scripts/           # Auxiliary tool scripts
├── dataset/               # Low-light datasets
├── pretrained_models/     # Pre-trained checkpoints
├── test.py                # Inference code
├── train.py               # Training code
├── metrics.py             # Evaluation metrics
└── environment.yml        # Conda environment config
```

---

## ⚙️ Environment Installation
### 1. Create Conda Environment
```bash
conda env create -f environment.yml
conda activate torch39
```

### 2. Install Dependencies
```bash
cd PyDiff
pip install -e .
cd ../BasicSR-light
pip install -e .
```

---

## 📂 Dataset Preparation
We use the public LOL datasets for training and evaluation:
- **LOL-v1**: 485 training pairs / 15 testing pairs
- **LOL-v2-real**: 689 training pairs / 100 testing pairs
- **LOL-v2-syn**: 689 training pairs / 100 testing pairs
  
🔗 **Baidu Netdisk**:  https://pan.baidu.com/s/1bRHaHVIwwaNtDVAI0iAVfQ?pwd=SADM

🔑 **Extract Code**: `SADM`

---

## 🚀 Quick Inference
```bash
python test.py -opt PyDiff/options/infer_v1.yaml
```
Enhanced results will be saved in the visualization folder.

## 🏋️ Training
Modify hyperparameters in `PyDiff/options/train_v1.yaml`, then run:
```bash
python train.py -opt PyDiff/options/train_v1.yaml
```

---

## 📊 Evaluation Metrics
We support mainstream low-light image evaluation metrics:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity)
- **LPIPS** (Learned Perceptual Similarity)

---

## 📥 Pre-trained Weights
The pre-trained weights for LOL-v1/LOLv2-real/LOLv2-syn are available:
🔗 **Baidu Netdisk**: https://pan.baidu.com/s/1HUDTolLpL4AXNI6BUxfF8Q?pwd=SADM
🔑 **Extract Code**: `SADM`

Put weights into `pretrained_models/lolv1/` for direct inference.

**Experimental Visualization Results**

All visual results in LOLv1, LOLv2_real, LOLv2_syn and ablation study are provided in the shared Baidu Netdisk file:
🔗 **Baidu Netdisk**: https://pan.baidu.com/s/1v2_dZnMrsee4rtqhTjayfA?pwd=SADM
🔑 **Extract Code**: `SADM`

---

## 📝 Citation
If you find this work useful for your research, please cite:
```bibtex
@article{liu2026single,
  title={Single-Stage Signal Attenuation Diffusion Model for Low-Light Image Enhancement and Denoising},
  author={Liu, Ying and Zhang, Junchao and Wu, Caiyun},
  journal={arXiv preprint arXiv:2604.05727},
  year={2026},
  doi={10.48550/arXiv.2604.05727}
}
```

---

## 🪪 License
This project is **only for academic research**. Commercial use is strictly prohibited.

## ⭐ Star
If this repository helps you, please give a star ⭐. Thank you for your support!
