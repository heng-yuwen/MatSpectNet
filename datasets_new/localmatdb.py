"""
This file implements the load and transform code for Local Material Database.
"""
import pickle
import pytorch_lightning as pl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os.path
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import Compose
from customise_pl.transforms import CommonCompose, init_transforms

lmd_color_plate = np.array([[119, 17, 17], [202, 198, 144], [186, 200, 238], [124, 143, 166], [89, 125, 49],
                  [16, 68, 16], [187, 129, 156], [208, 206, 72], [98, 39, 69], [102, 102, 102],
                  [76, 74, 95], [16, 16, 68], [68, 65, 38], [117, 214, 70], [221, 67, 72],
                  [92, 133, 119], [0, 0, 0]])

def lmd_labels_to_color(labels):
    """ Convert netcat labels to a color-mapped image """
    image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    image[:, :, :] = lmd_color_plate[labels]
    return image

class LocalMatDataModule(pl.LightningDataModule):
    def __init__(self, data_root, image_folder="_resized", mask_folder="masks_resized", batch_size=4, num_workers=8,
                 pin_memory=True, partition_idx=1,
                 train_image_transform=None,
                 train_common_transform=None,
                 valid_image_transform=None,
                 valid_common_transform=None,
                 test_image_transform=None,
                 test_common_transform=None):
        super().__init__()
        self.train_files = None
        self.test_files = None
        self.valid_files = None
        self.train_names = None
        self.image_path = None
        self.mask_path = None
        self.data_root = data_root
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.partition_idx = partition_idx
        self.train_image_transform = Compose(init_transforms(train_image_transform))
        self.train_common_transform = CommonCompose(init_transforms(train_common_transform))
        self.test_image_transform = Compose(init_transforms(test_image_transform))
        self.test_common_transform = CommonCompose(init_transforms(test_common_transform))

        if valid_image_transform is None:
            self.valid_image_transform = Compose(init_transforms(test_image_transform))
        else:
            self.valid_image_transform = Compose(init_transforms(valid_image_transform))
        if valid_common_transform is None:
            self.valid_common_transform = CommonCompose(init_transforms(test_common_transform))
        else:
            self.valid_common_transform = CommonCompose(init_transforms(valid_common_transform))

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        # make assignments here ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``, include transforms and dataset
        # called on every process in DDP
        self.image_path = os.path.join(self.data_root, self.image_folder)
        self.mask_path = os.path.join(self.data_root, self.mask_folder)
        

    def train_dataloader(self):
        train_split = LocalMatDataset(os.path.join(self.data_root, f"train_{self.partition_idx}"), self.data_root, self.image_folder, self.mask_folder,
                                      image_transform=self.train_image_transform,
                                      common_transform=self.train_common_transform)
        return DataLoader(train_split, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        val_split = LocalMatDataset(os.path.join(self.data_root, f"validate_{self.partition_idx}"), self.data_root, self.image_folder, self.mask_folder,
                                    image_transform=self.valid_image_transform,
                                    common_transform=self.valid_common_transform)
        return DataLoader(val_split, batch_size=1, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        test_split = LocalMatDataset(os.path.join(self.data_root, f"test_{self.partition_idx}"), self.data_root, self.image_folder, self.mask_folder,
                                     image_transform=self.test_image_transform,
                                     common_transform=self.test_common_transform)
        return DataLoader(test_split, batch_size=1, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)
    
    def pred_dataloader(self):
        image_files = ["2009_000212.png", "COCO_train2014_000000111022.png", "COCO_train2014_000000203887.png", "COCO_train2014_000000341917.png", "COCO_train2014_000000377583.png", "COCO_train2014_000000417814.png", "COCO_train2014_000000510577.png", "COCO_train2014_000000555211.png"]
        pred_split = LocalMatDataset(image_files, self.data_root, self.image_folder, "test_indoor",
                                     image_transform=self.test_image_transform,
                                     common_transform=self.test_common_transform)
        return DataLoader(pred_split, batch_size=1, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def teardown(self, stage):
        # clean up after fit or test
        # called on every process in DDP
        pass


class LocalMatDataset(Dataset):
    CLASSES = [
        "asphalt",
        "ceramic",
        "concrete",
        "fabric",
        "foliage",
        "food",
        "glass",
        "metal",
        "paper",
        "plaster",
        "plastic",
        "rubber",
        "soil",
        "stone",
        "water",
        "wood"
    ]

    def __init__(self, image_partition_file, data_root, image_folder="images", mask_folder="masks", image_transform=None,
                 common_transform=None):
        super().__init__()
        self.image_files = list(pickle.load(open(image_partition_file, "rb"))) if isinstance(image_partition_file, str) else image_partition_file
        self.data_root = data_root
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_transform = image_transform
        self.common_transform = common_transform

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, self.image_folder, self.image_files[index][:-4] + ".png")
        mask_path = os.path.join(self.data_root, self.mask_folder, self.image_files[index][:-4] + ".png")
        image = pil_to_tensor(Image.open(img_path).convert('RGB'))
        mask = pil_to_tensor(Image.open(mask_path))

        # used to recover the rgb.
        image_conf = {"image_max": image.max(), "image_min": image.min(), "filename": self.image_files[index][:-4]}

        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.common_transform is not None:
            image, mask = self.common_transform(image, mask)
        return image, mask.squeeze(), image_conf

    def __len__(self):
        return len(self.image_files)
