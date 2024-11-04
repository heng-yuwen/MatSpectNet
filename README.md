# MatSpectNet: A Physically-Constrained Hyperspectral Reconstruction Network for Domain-Aware Material Segmentation
[![arXiv](https://img.shields.io/badge/arxiv-paper-179bd3)](https://arxiv.org/abs/2307.11466)

Yuwen Heng, Yihong Wu, Srinandan Dasmahapatra, Hansung Kim

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
The datasset used for spectral recovery task is the ARAD_1K, from the NTIRE 2022 Challenge on Spectral Reconstruction from RGB, at https://codalab.lisn.upsaclay.fr/competitions/721. **Some of the hyperspectral images contain zero elements. They should be deleted first for MST++.**

The directory structure in ./data should be organised as follows: 

ARAD_1K\
|-- NTIRE2022_spectral\
|-- Test_RGB\
|-- Train_RGB\
|-- Train_spectral\
|-- Valid_RGB\
|-- Valid_spectral

### Spectraldb Dataset
The spectral dataset can be downloaded from https://github.com/C38C/SpectralDB, which is already included in data/spectraldb. The main file is the shape_metrics.npy, which transforms the spectral profile into a shape description, as described in the main paper. The implementation of the transformation is in the file convert_hsi_shape.py


## ConfigLightning for MatSpectNet
This repo use config to parse the training configs, with pytorch-lightning as the training framework.

Please download the checkpoints, and put them in the folder "checkpoints" first.

MatSpectNet checkpoint:
https://drive.google.com/file/d/1bk0bMVLipnmUv9Ttl8GZtXY4PB61_stZ/view?usp=drive_link

Swin backbone checkpoint:
https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window8_256.pth

Use segmentation experiment:
```python
python main.py test --config configs/matspectnet/test.yaml # test on test split of LMD
```
The code is configured to train with 8 NVIDIA GeForce RTX 3090 GPUs.  
