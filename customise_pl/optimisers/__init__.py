"""Support the import of optimisers from torch with string.
"""
from torch import optim


def build_optimizer(params, cfg):
    assert "type" in cfg, "Must specify which optimiser to use in type"
    return getattr(optim, cfg.pop("type"))(params=params, **cfg)
