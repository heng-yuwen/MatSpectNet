"""ARAD_1K dataset and its loader.
"""
import os
from glob import glob
import random

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image

from customise_pl.transforms import CommonCompose
from customise_pl.transforms import init_transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor


class SpectralRecoveryDataModule(pl.LightningDataModule):
    def __init__(self, data_root, train_rgb_folder="Train_RGB", noisefree_train_rgb_folder=None,
                 train_spectral_folder="Train_spectral",
                 valid_rgb_folder="Valid_RGB", noisefree_valid_rgb_folder=None, valid_spectral_folder="Valid_spectral",
                 test_rgb_folder="Test_RGB", num_workers=8, batch_size=8, pin_memory=True, train_transform=None,
                 valid_transform=None, test_transform=None, random_select=False):
        super().__init__()
        self.test_files = None
        self.valid_files = None
        self.train_files = None
        self.data_root = data_root
        self.train_rgb_folder = train_rgb_folder
        self.noisefree_train_rgb_folder = noisefree_train_rgb_folder
        self.train_spectral_folder = train_spectral_folder
        self.valid_rgb_folder = valid_rgb_folder
        self.noisefree_valid_rgb_folder = noisefree_valid_rgb_folder
        self.valid_spectral_folder = valid_spectral_folder
        self.test_rgb_folder = test_rgb_folder
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.train_transform = CommonCompose(init_transforms(train_transform))
        self.test_transform = CommonCompose(init_transforms(test_transform))

        self.random_select = random_select

        if valid_transform is None:
            self.valid_transform = CommonCompose(init_transforms(test_transform))
        else:
            self.valid_transform = CommonCompose(init_transforms(valid_transform))

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        self.train_files = [os.path.basename(file)[:-4] for file in
                            glob(os.path.join(self.data_root, self.train_rgb_folder, "*.jpg"))]
        self.valid_files = [os.path.basename(file)[:-4] for file in
                            glob(os.path.join(self.data_root, self.valid_rgb_folder, "*.jpg"))]
        self.test_files = [os.path.basename(file)[:-4] for file in
                           glob(os.path.join(self.data_root, self.test_rgb_folder, "*.jpg"))]

    def train_dataloader(self):
        train_split = ARAD1KDataset(self.train_files, self.data_root, self.train_rgb_folder,
                                    self.noisefree_train_rgb_folder,
                                    self.train_spectral_folder, image_transform=self.train_transform,
                                    random_select=self.random_select)
        return DataLoader(train_split, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, drop_last=True)

    def val_dataloader(self):
        val_split = ARAD1KDataset(self.valid_files, self.data_root, self.valid_rgb_folder,
                                  self.noisefree_valid_rgb_folder,
                                  self.valid_spectral_folder, image_transform=self.valid_transform)
        return DataLoader(val_split, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, drop_last=True)

    def test_dataloader(self):
        test_split = ARAD1KDataset(self.test_files, self.data_root, self.test_rgb_folder, None, None,
                                   image_transform=self.test_transform)
        return DataLoader(test_split, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def teardown(self, stage):
        # clean up after fit or test
        # called on every process in DDP
        pass


class ARAD1KDataset(Dataset):
    def __init__(self, image_files, data_root, rgb_folder="Train_RGB", noisefree_rgb_folder=None,
                 spectral_folder="Train_spectral", image_transform=None, random_select=False):
        super().__init__()
        self.image_files = image_files
        self.data_root = data_root
        self.rgb_folder = rgb_folder
        self.spectral_folder = spectral_folder
        self.image_transform = image_transform
        self.noisefree_rgb_folder = noisefree_rgb_folder
        self.random_select = random_select

    def __getitem__(self, index):
        if self.random_select:
            index = random.randint(0, len(self) - 1)
        rgb_path = os.path.join(self.data_root, self.rgb_folder, self.image_files[index] + ".jpg")
        rgb = pil_to_tensor(Image.open(rgb_path).convert('RGB'))
        if self.noisefree_rgb_folder is not None:
            noisefree_rgb_path = os.path.join(self.data_root, self.noisefree_rgb_folder,
                                              self.image_files[index] + ".png")
            noisefree_rgb = pil_to_tensor(Image.open(noisefree_rgb_path).convert('RGB'))
        else:
            noisefree_rgb = None
        if self.spectral_folder is not None:
            spectral_path = os.path.join(self.data_root, self.spectral_folder, self.image_files[index] + ".mat")
            try:
                with h5py.File(spectral_path, 'r') as mat:
                    spectral = torch.as_tensor(np.transpose(np.array(mat['cube'], dtype="float32"), [0, 2, 1]))
                if (spectral == 0).any():
                    print(spectral_path)
            except:
                raise EOFError("{} is corrupted".format(spectral_path))
        else:
            spectral = None

        if self.image_transform is not None and self.noisefree_rgb_folder is None:
            rgb, spectral = self.image_transform(rgb, spectral)
            if len(rgb) == 1:
                rgb = rgb[0]
        elif self.image_transform is not None and self.noisefree_rgb_folder is not None:
            (rgb, noisefree_rgb), spectral = self.image_transform([rgb, noisefree_rgb], spectral)

        data = [rgb]
        if noisefree_rgb is not None:
            data.append(noisefree_rgb)
        if spectral is not None:
            data.append(spectral)

        return data

    def __len__(self):
        return len(self.image_files)
