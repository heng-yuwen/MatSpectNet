from collections import OrderedDict

import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from models import classification_models
from models import segmentation_models
from timm.models.layers import trunc_normal_
from models.segmentation_models.dpglt_single_branch import BasicLayer


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, scale_rate=2):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(scale_rate ** 2 * dim, scale_rate * dim, bias=False)
        self.norm = norm_layer(scale_rate ** 2 * dim)
        self.scale_rate = scale_rate

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x_list = []
        for i in range(self.scale_rate):
            for j in range(self.scale_rate):
                x_list.append(x[:, i::self.scale_rate, j::self.scale_rate, :])  # B H/2 W/2 C

        x = torch.cat(x_list, -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, self.scale_rate ** 2 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class EnlargeChannel(nn.Module):
    def __init__(self, in_chans, out_chans, scale_rate, norm_layer=nn.LayerNorm):
        super().__init__()
        self.out_chans = out_chans
        self.scale_rate = scale_rate
        if scale_rate > 1:
            self.enlarge_conv = PatchMerging(in_chans, scale_rate=scale_rate)
            self.norm_enlarge = norm_layer(out_chans)

    def forward(self, x):
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        x = self.enlarge_conv(x, H, W)
        x = self.norm_enlarge(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.out_chans, H // self.scale_rate, W // self.scale_rate)

        return x


class CrossResolutionAttention(pl.LightningModule):
    def __init__(self, DPATCH_INNER_3x3=True,
                 DPATCH_DROPOUT=False,
                 patch_numbers=4,
                 out_chans=96):
        super(CrossResolutionAttention, self).__init__()
        in_chans = 768
        mid_chans = 384
        self.reduce_chans = False

        od = OrderedDict([('conv0', nn.Conv2d(in_chans, mid_chans, kernel_size=3,
                                              padding=1, bias=False)),
                          ('bn0', nn.BatchNorm2d(mid_chans)),
                          ('re0', nn.ReLU(inplace=True))])

        if DPATCH_INNER_3x3:
            od['conv1'] = nn.Conv2d(mid_chans, mid_chans, kernel_size=3, padding=1,
                                    bias=False)
            od['bn1'] = nn.BatchNorm2d(mid_chans)
            od['re1'] = nn.ReLU(inplace=True)

        if DPATCH_DROPOUT:
            od['drop'] = nn.Dropout(0.5)

        od['conv2'] = nn.Conv2d(mid_chans, patch_numbers, kernel_size=1, bias=False)
        od['sig'] = nn.Softmax(dim=1)  # make sure the weights sum to 1 at each position.

        self.attn_head = nn.Sequential(od)

        # channel reduction conv for the global feature map after the dynamic patch stage.
        if self.reduce_chans:
            self.chans_reduce = nn.Sequential(
                OrderedDict([("conv_reduce1", nn.Conv2d(in_chans, mid_chans, kernel_size=1, stride=1)),
                             ("bn_reduce1", nn.BatchNorm2d(mid_chans)),
                             ("re_reduce1", nn.ReLU(inplace=True)),
                             ("conv_reduce2", nn.Conv2d(mid_chans, out_chans, kernel_size=1, stride=1))]))

    def forward(self, map4):
        att = self.attn_head(map4)
        return att


def reformat_checkpoint_v2(new_backbone, official_checkpoint):
    # Only work for patch size >= 256
    checkpoint = torch.load(official_checkpoint, map_location="cpu")["model"]
    reformatted_checkpoint = new_backbone.state_dict()
    reinitialsed_layers = ["layers.0.blocks.1.attn_mask",
                           "layers.1.blocks.1.attn_mask", "layers.2.blocks.1.attn_mask",
                           "layers.2.blocks.3.attn_mask", "layers.2.blocks.5.attn_mask"]

    for layername in reinitialsed_layers:
        official_weight = checkpoint[layername]
        if len(official_weight.shape) == 4:
            C = official_weight.size(1)
            reformatted_checkpoint[layername][:, :C, :, :] = official_weight
            checkpoint[layername] = reformatted_checkpoint[layername]
        elif len(official_weight.shape) == 3:
            C = official_weight.size(0)
            reformatted_checkpoint[layername][:C, :, :] = official_weight
            checkpoint[layername] = reformatted_checkpoint[layername]
    # torch.save(checkpoint, "swinv2_tiny_patch4_window8_512.pth")
    return checkpoint


def reformat_checkpoint_v1(new_backbone, official_checkpoint):
    checkpoint = torch.load(official_checkpoint, map_location="cpu")["model"]
    reformatted_checkpoint = new_backbone.state_dict()
    reinitialsed_layers = ["patch_embed.proj.weight"]

    for layername in reinitialsed_layers:
        official_weight = checkpoint[layername]
        if len(official_weight.shape) == 4:
            C = official_weight.size(1)
            reformatted_checkpoint[layername][:, :C, :, :] = official_weight
            checkpoint[layername] = reformatted_checkpoint[layername]
        elif len(official_weight.shape) == 3:
            C = official_weight.size(0)
            reformatted_checkpoint[layername][:C, :, :] = official_weight
            checkpoint[layername] = reformatted_checkpoint[layername]
    # torch.save(checkpoint, "swinv2_tiny_patch4_window8_512.pth")
    return checkpoint


class DBAT(pl.LightningModule):
    def __init__(self, in_chans=3, backbone_checkpoint=None, image_size=512, backbone_model="swinv2"):
        super(DBAT, self).__init__()
        self.backbone = classification_models.get_models(model_name=backbone_model, extract_features=True,
                                                         in_chans=in_chans,
                                                         img_size=image_size)
        self.cross_resolution_attention = CrossResolutionAttention()
        self.out_channels = [3, 96, 192, 384, 768]

        # enlarge block for the global features
        self.enlarge_4x = EnlargeChannel(in_chans=self.out_channels[1], out_chans=self.out_channels[4],
                                         scale_rate=32 // 4)
        self.enlarge_8x = EnlargeChannel(in_chans=self.out_channels[2], out_chans=self.out_channels[4],
                                         scale_rate=32 // 8)
        self.enlarge_16x = EnlargeChannel(in_chans=self.out_channels[3], out_chans=self.out_channels[4],
                                          scale_rate=32 // 16)

        dpr = [x.item() for x in torch.linspace(0, 0.3, 2 * 4)]
        # GLTransformer block
        self.gltrans_32x = BasicLayer(dim=self.out_channels[4], num_heads=32, H=image_size // 32,
                                      W=image_size // 32, window_size=8, depth=2, mlp_ratio=4.,
                                      qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                                      drop_path=dpr[6:8])

        self.init_weights()

        # load checkpoint for the backbone
        if backbone_model == "swinv2":
            checkpoint = reformat_checkpoint_v2(self.backbone,
                                                backbone_checkpoint)  # run this when the in_chans changes
        elif backbone_model == "swinv1":
            checkpoint = reformat_checkpoint_v1(self.backbone, backbone_checkpoint)
        self.backbone.load_state_dict(checkpoint, strict=False)

    def forward(self, image):
        backbone_features = self.backbone(image)
        attns = self.cross_resolution_attention(backbone_features[-1])
        _, C, H, W = attns.size()

        resized_patch_features = [self.enlarge_4x(backbone_features[1]), self.enlarge_8x(backbone_features[2]),
                                  self.enlarge_16x(backbone_features[3]), backbone_features[4]]
        assert C == len(resized_patch_features), "The attention channel number does not match the patch feature number"

        # sum the weighted patch feature maps
        for idx, patch_feature in enumerate(resized_patch_features):
            resized_patch_features[idx] = torch.mul(patch_feature, torch.unsqueeze(attns[:, idx, :, :], dim=1))

        local_features = torch.sum(torch.stack(resized_patch_features), dim=0)
        features_32x = self.gltrans_32x(local_features, backbone_features[-1])  # B * H*W * C

        return [backbone_features[0], backbone_features[1], backbone_features[2], backbone_features[3], features_32x]

    def init_weights(self):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)


class DBATSeg(pl.LightningModule):
    def __init__(self, backbone_checkpoint, image_size, classes=16, backbone_model="swinv2"):
        super(DBATSeg, self).__init__()
        dbat = DBAT(backbone_checkpoint=backbone_checkpoint, image_size=image_size, backbone_model=backbone_model)
        decoder = segmentation_models.get_models(encoder=dbat, model_name="FPN", encoder_depth=5, classes=classes,
                                                 encoder_weights=None, train_ops=False,
                                                 train_ops_mat_only=False)
        self.model = decoder

    def forward(self, x, filtered_hsi=None):
        x = self.model(x, filtered_hsi)
        # print("here")
        return x
