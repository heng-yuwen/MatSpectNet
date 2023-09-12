from torchvision.transforms import *
from .segment_transforms import *
from .spectral_recovery_transforms import *


class CommonCompose(Compose):
    def __init__(self, transforms):
        super().__init__(transforms)

    def __call__(self, image1, image2):
        for t in self.transforms:
            image1, image2 = t(image1, image2)
        return image1, image2


def init_class(transform):
    class_path = transform["class_path"]
    class_ = eval(class_path)

    if "init_args" in transform:
        init_args = transform["init_args"]
        return class_(**init_args)

    else:
        return class_()


def init_transforms(transforms):
    if transforms is None:
        return None
    initialised = [init_class(transform) for transform in transforms]
    return initialised
