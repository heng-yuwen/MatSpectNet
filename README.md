# MatSpectNet: A Physically-Constrained Hyperspectral Reconstruction Network for Domain-Aware Material Segmentation
[![arXiv](https://img.shields.io/badge/arxiv-paper-179bd3)](https://arxiv.org/abs/2307.11466)

Yuwen Heng, Yihong Wu, Jiawen Chen, Srinandan Dasmahapatra, Hansung Kim

## Environment
To install the required environment, please use the command below:
```
pip install -r requirements.txt
```

## Dataset
The dataset used in this code is the Local Material Dataset (LMD) from the Kyoto University Computer Vision Lab, please download it at the official website: https://vision.ist.i.kyoto-u.ac.jp/codeanddata/localmatdb/

## ConfigLightning for MatSpectNet
This repo use config to parse the training configs, with pytorch-lightning as the training framework.
Use segmentation experiment:
```python

python train.py fit --config configs/spectral_recovery/spectral_config.yaml # pre-train the spectral recovery network S(x)
python train.py fit --config configs/matspectnet/train.yaml # fit on train split of LMD.
python train.py test --config configs/matspectnet/test.yaml # test on test split of LMD
```
The code is configured to train with 8 NVIDIA GeForce RTX 3090 GPUs.  