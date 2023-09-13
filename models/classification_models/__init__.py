from .swinv2 import SwinTransformerV2
from .swinv1 import SwinTransformer

__models__ = {
    "swinv1": SwinTransformer,
    "swinv2": SwinTransformerV2
}

def get_models(**kwargs):
    if "model_name" in kwargs:
        model_name = kwargs.pop("model_name")
        assert model_name in __models__, f"The model {model_name} does not exist"
        return __models__[model_name](**kwargs)
