import pytorch_lightning as pl
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import random


class SpectralNorm(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, rgb, spectral):
        if not isinstance(rgb, list):
            rgb = [rgb]
        for i in range(len(rgb)):
            rgb[i] = (rgb[i] - rgb[i].min()) / (rgb[i].max() - rgb[i].min())

        return rgb, spectral


class SpectralRotateFlip(pl.LightningModule):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, rgb, spectral):
        if not isinstance(rgb, list):
            rgb = [rgb]
        for j in range(random.randint(0, 3)):
            for i in range(len(rgb)):
                rgb[i] = F.rotate(rgb[i], 90)
            if spectral is not None:
                spectral = F.rotate(spectral, 90)
        # Random vertical Flip
        if random.random() > self.p:
            for i in range(len(rgb)):
                rgb[i] = F.vflip(rgb[i])
            if spectral is not None:
                spectral = F.vflip(spectral)
        # Random horizontal Flip
        if random.random() > self.p:
            for i in range(len(rgb)):
                rgb[i] = F.hflip(rgb[i])
            if spectral is not None:
                spectral = F.hflip(spectral)
        return rgb, spectral


class SpectralRandomCrop(pl.LightningModule):
    """Use this after the SpectralNorm class forward"""
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, rgb, spectral):
        if not isinstance(rgb, list):
            rgb = [rgb]
        i, j, h, w = transforms.RandomCrop.get_params(rgb[0], self.patch_size)
        for i in range(len(rgb)):
            rgb[i] = F.crop(rgb[i], i, j, h, w)
        if spectral is not None:
            spectral = F.crop(spectral, i, j, h, w)

        return rgb, spectral
