"""Support the import of schedulers with strings.
"""
from typing import Dict, Union
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR
from .warmup_schedulers import PolyLRScheduler
from .linear_scheduler import LinearLR

__all__ = {"PolyLRScheduler": PolyLRScheduler,
           "CosineAnnealingLR": CosineAnnealingLR,
           "LinearLR": LinearLR,
           "CyclicLR": CyclicLR}


def build_scheduler(optimizer, cfg: Dict, num_epochs: int, num_training_steps: Union[int, None] = None):
    """
    Build a scheduler.
    
    Args:
        optimizer: the name of the optimizer
        cfg: scheduler configuration
        num_epochs: number of training epochs
        num_training_steps: the training steps per epoch
    
    Returns:
        None or a scheduler
    
    """
    if not isinstance(cfg, Dict):
        return None
    assert "type" in cfg, "Scheduler type is not specified"
    if "by_iteration" in cfg and cfg.pop("by_iteration"):
        total_iteration = num_epochs * num_training_steps
        scheduler = __all__[cfg.pop("type")](optimizer, total_iteration, **cfg)
        scheduler.by_iteration = True
        return scheduler
    elif "by_stage" in cfg and cfg.pop("by_stage"):
        scheduler = __all__[cfg.pop("type")](optimizer, num_epochs=num_epochs, **cfg)
        scheduler.by_stage = True
        return scheduler
    elif cfg["type"] != "CyclicLR":
        scheduler = __all__[cfg.pop("type")](optimizer, num_epochs=num_epochs, **cfg)
        scheduler.by_iteration = False
        return scheduler
    else:
        start_epoch = cfg.pop("start_epoch")
        scheduler = __all__[cfg.pop("type")](optimizer, **cfg)
        scheduler.by_iteration = False
        scheduler.start_epoch = start_epoch
        return scheduler
