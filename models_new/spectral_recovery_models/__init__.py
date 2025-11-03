import torch
# from .edsr import EDSR
# from .HDNet import HDNet
# from .hinet import HINet
# from .hrnet import SGN
# from .HSCNN_Plus import HSCNN_Plus
# from .MIRNet import MIRNet
# from .MPRNet import MPRNet
# from .MST import MST
from .MST_Plus_Plus import MST_Plus_Plus
# from .Restormer import Restormer
# from .AWAN import AWAN
# from .Reverser import HSItoRGBReverser
# from .image_pool import ImagePool
# from .Reverserv2 import HSItoRGBReverserV2
# from .MST_Plus_Plus_v2 import MST_Plus_PlusV2


def get_models(method, pretrained_model_path=None, **kwargs):
    # if method == 'mirnet':
    #     model = MIRNet(n_RRG=3, n_MSRB=1, height=3, width=1)
    if method == 'mst_plus_plus':
        model = MST_Plus_Plus()
    # elif method == 'mst_plus_plusv2':
    #     model = MST_Plus_PlusV2()
    # elif method == 'mst':
    #     model = MST(dim=31, stage=2, num_blocks=[4, 7, 5])
    # elif method == 'hinet':
    #     model = HINet(depth=4)
    # elif method == 'mprnet':
    #     model = MPRNet(num_cab=4)
    # elif method == 'restormer':
    #     model = Restormer()
    # elif method == 'edsr':
    #     model = EDSR()
    # elif method == 'hdnet':
    #     model = HDNet()
    # elif method == 'hrnet':
    #     model = SGN()
    # elif method == 'hscnn_plus':
    #     model = HSCNN_Plus()
    # elif method == 'awan':
    #     model = AWAN()
    # elif method == "HSItoRGBReverser":
    #     model = HSItoRGBReverser(rgb_camera_qe_path=kwargs["rgb_camera_qe_path"], use_trainable_rgb_filters=kwargs["use_trainable_rgb_filters"])
    # elif method == "HSItoRGBReverserV2":
    #     model = HSItoRGBReverserV2(rgb_camera_qe_path=kwargs["rgb_camera_qe_path"], use_trainable_rgb_filters=kwargs["use_trainable_rgb_filters"], use_l_band=kwargs["use_l_band"])
    else:
        raise NotImplementedError(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        model.load_state_dict({k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()},
                              strict=True)
    return model
