from collections import OrderedDict

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.checkpoint as checkpoint
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from models.classification_models.swinv1 import SwinTransformerBlock
from segmentation_models_pytorch.encoders import get_encoder


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GLAttention(nn.Module):
    """ Global Local attention, designed to compensate the local material features and the global contextual features.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super(GLAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.local_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.global_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.local_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, local_features, global_features, mask=None):
        """ Forward function
        Args:
            local_features: input local features with shape (num_windows*B, N, C)
            global_features: input global features with shape (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows*B, N, C)or None
        """
        B_, N, C = local_features.shape  # nB, N = H*W

        assert local_features.shape == global_features.shape, "Global feature doesn't match with local feature."
        local_k = torch.squeeze(
            self.local_k(local_features).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4))
        global_q = torch.squeeze(
            self.global_q(global_features).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                          4))
        local_v = torch.squeeze(
            self.local_v(local_features).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                          4))

        global_q = global_q * self.scale
        attn = (global_q @ local_k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ local_v).transpose(1, 2).contiguous().reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GLTransformerBlock(nn.Module):
    """ GL Transformer Block (GLAttention + FFN).
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            window_size (int): size of attention window.
            shift_size (int): Shift size for SW-MSA.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
            drop (float, optional): Dropout rate. Default: 0.0
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            drop_path (float, optional): Stochastic depth rate. Default: 0.0
            act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """

    def __init__(self, dim, num_heads, window_size, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm_local = norm_layer(dim)
        self.norm_global = norm_layer(dim)

        self.attn = GLAttention(
            dim, num_heads=num_heads, window_size=to_2tuple(self.window_size),
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, local_features, global_features, mask_matrix=None):
        """ Forward function.
        Args:
            local_features, global_features: Input feature, tensor size (B, C, H, W).
            mask_matrix: Attention mask for transformer.
        """
        local_features = local_features.flatten(2).transpose(1, 2)  # transpose to normalise the channel dim only
        global_features = global_features.flatten(2).transpose(1, 2)  # transpose to normalise the channel dim only
        B, L, C = local_features.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size, L is {}, H, W is {}. the local features have shape {}, global features have shape {}".format(
            L, (H, W), local_features.size(), global_features.size())

        # local residual
        shortcut = global_features

        local_features = self.norm_local(local_features)
        global_features = self.norm_global(global_features)
        local_features = local_features.view(B, H, W, C)
        global_features = global_features.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        local_features = F.pad(local_features, (0, 0, pad_l, pad_r, pad_t, pad_b))
        global_features = F.pad(global_features, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = local_features.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_local_features = torch.roll(local_features, shifts=(-self.shift_size, -self.shift_size),
                                                dims=(1, 2))
            shifted_global_features = torch.roll(global_features, shifts=(-self.shift_size, -self.shift_size),
                                                 dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_local_features = local_features
            shifted_global_features = global_features
            attn_mask = None

        # partition windows
        local_features_windows = window_partition(shifted_local_features,
                                                  self.window_size)  # nW*B, window_size, window_size, C
        global_features_windows = window_partition(shifted_global_features,
                                                   self.window_size)  # nW*B, window_size, window_size, C
        local_features_windows = local_features_windows.view(-1, self.window_size * self.window_size,
                                                             C)  # nW*B, window_size*window_size, C
        global_features_windows = global_features_windows.view(-1, self.window_size * self.window_size,
                                                               C)  # nW*B, window_size*window_size, C

        # GLAttention with W-MSA/SW-MSA
        attn_windows = self.attn(local_features_windows, global_features_windows,
                                 mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN to combine local with global from local attention
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm(x)))

        # x = x.transpose(1, 2).contiguous().view(B, C, H, W)

        return x


class pointwise_conv(nn.Module):
    def __init__(self, nin, nout):
        super(pointwise_conv, self).__init__()
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(nout)

    def forward(self, x):
        x = self.pointwise(x)
        C, Wh, Ww = x.size(1), x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        out = x.transpose(1, 2).contiguous().view(-1, C, Wh, Ww)
        return out


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
        x = x.transpose(1, 2).contiguous().view(-1, self.out_chans, H//self.scale_rate, W//self.scale_rate)

        return x


class GlobalAttConv(nn.Module):
    """ The conv branch to extract features from the full image, and calculate the weights across different patch size
        features
    Args:
        DPATCH_INNER_3x3 (bool): use the intermidiate 3x3 conv or not. Default: True.
        DPATCH_DROPOUT (bool): use the dropout or not at the attention head. Default: False.
        patch_numbers (int): the number of weights needed for patch feature maps, equal to the patch number. Default: 7
        out_chans (int): the output channels of the combined patch feature maps. Default: 96.
    """

    def __init__(self,
                 DPATCH_INNER_3x3=True,
                 DPATCH_DROPOUT=False,
                 patch_numbers=4,
                 out_chans=96,
                 net="resnet"):
        super(GlobalAttConv, self).__init__()

        if net == "resnet":
            self.glconv = get_encoder("resnet50", weights=None, in_channels=3, depth=5)
            in_chans = 2048
            mid_chans = 512
            self.reduce_chans = True
        else:
            self.glconv = get_encoder(net, in_channels=3, depth=5, weights=None)
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

    def forward(self, x):
        x = self.glconv(x)
        att = self.attn_head(x[-1])

        if self.reduce_chans:
            x[-1] = self.chans_reduce(x[-1])

        return att, x


class BasicLayer(nn.Module):
    """ A basic GL Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 16.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 H, W,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.H = H
        self.W = W
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.gltrans = GLTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer)

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(1, depth)])

        self.norm = norm_layer(dim)

    def forward(self, local_features, global_features):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H, W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(self.H / self.window_size)) * self.window_size
        Wp = int(np.ceil(self.W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=local_features.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        B, C, _, _ = local_features.size()
        self.gltrans.H, self.gltrans.W = self.H, self.W
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.gltrans, local_features, global_features, attn_mask)
        else:
            x = self.gltrans(local_features, global_features, attn_mask)

        for blk in self.blocks:
            blk.H, blk.W = self.H, self.W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

            x = self.norm(x)
            x = x.transpose(1, 2).contiguous().view(B, C, self.H, self.W)
        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm, scale_rate=2):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(scale_rate**2 * dim, scale_rate * dim, bias=False)
        self.norm = norm_layer(scale_rate**2 * dim)
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

        # x0 = x[:, 0::self.scale_rate, 0::self.scale_rate, :]  # B H/2 W/2 C
        # x1 = x[:, 1::self.scale_rate, 0::self.scale_rate, :]  # B H/2 W/2 C
        # x2 = x[:, 0::self.scale_rate, 1::self.scale_rate, :]  # B H/2 W/2 C
        # x3 = x[:, 1::self.scale_rate, 1::self.scale_rate, :]  # B H/2 W/2 C

        x = torch.cat(x_list, -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, self.scale_rate**2 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
