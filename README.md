# MatSpectNet: A Physically-Constrained Hyperspectral Reconstruction Network for Domain-Aware Material Segmentation
[![arXiv](https://img.shields.io/badge/arxiv-paper-179bd3)](https://arxiv.org/abs/2307.11466)

Yuwen Heng, Yihong Wu, Jiawen Chen, Srinandan Dasmahapatra, Hansung Kim

## Environment
To install the required environment, please use the command below:
```
pip install -r requirements.txt
```

## Datasets
### Material Segmentation Dataset
The dataset used for material segmentation is the Local Material Dataset (LMD) from the Kyoto University Computer Vision Lab, please download it at the official website: https://vision.ist.i.kyoto-u.ac.jp/codeanddata/localmatdb/. Please unpack the downloaded dataset complete_localmatdb.tgz to the folder ./data, and move the images and masks folders to ./data/localmatdb.

Then, run the following code to generate the ground truth masks.
```python
python convert_lmd.py
```

### Spectral Recovery Dataset
The datasset used for spectral recovery task is the ARAD_1K, from the NTIRE 2022 Challenge on Spectral Reconstruction from RGB, at https://codalab.lisn.upsaclay.fr/competitions/721

### Spectraldb Dataset
The spectral dataset can be downloaded from https://github.com/C38C/SpectralDB, which is already included in data/spectraldb

## ConfigLightning for MatSpectNet
This repo use config to parse the training configs, with pytorch-lightning as the training framework.
Use segmentation experiment:
```python

python main.py fit --config configs/spectral_recovery/spectral_config.yaml # pre-train the spectral recovery network S(x)
python main.py fit --config configs/matspectnet/train.yaml # fit on train split of LMD.
python main.py test --config configs/matspectnet/test.yaml # test on test split of LMD
```
The code is configured to train with 8 NVIDIA GeForce RTX 3090 GPUs.  