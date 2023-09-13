"""The dataloaders to load and process multiple datasets.
"""
from .mask_segment import XJ3SegmentDataModule
from .arad1k import SpectralRecoveryDataModule
from .localmatdb import LocalMatDataModule
from .unaligned_lmd_arad1k import UnalignedLMDARAD1K

__all__ = {"XJ3SegmentDataModule": XJ3SegmentDataModule,
           "SpectralRecoveryDataModule": SpectralRecoveryDataModule,
           "LocalMatDataModule": LocalMatDataModule,
           "UnalignedLMDARAD1K": UnalignedLMDARAD1K}

def get_dataset(dataset_name):
    assert dataset_name in __all__, f"The dataset {dataset_name} is not supported!"
    return __all__[dataset_name]
