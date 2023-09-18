import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip_conv(skip)
        x = x + skip
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(
                    policy
                )
            )
        self.policy = policy

    def forward(self, x):
        if self.policy == 'add':
            return sum(x)
        elif self.policy == 'cat':
            return torch.cat(x, dim=1)
        else:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy)
            )


class FPNDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            encoder_depth=5,
            pyramid_channels=256,
            segmentation_channels=128,
            dropout=0.2,
            merge_policy="add",
            mode = 1
    ):
        super().__init__()

        self.encoder_depth = encoder_depth
        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError("Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))

        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[:encoder_depth + 1]

        # last one
        self.p_last = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
        self.p_list = nn.ModuleList()
        for i in range(1, encoder_depth-1):
            self.p_list.append(FPNBlock(pyramid_channels, encoder_channels[i]))

        self.seg_blocks = nn.ModuleList([
            SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
            for n_upsamples in list(range(encoder_depth-1))[::-1]
        ])

        self.merge = MergeBlock(merge_policy)
        self.merge_hsi = MergeBlock("cat")  
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        self.out_channels = self.out_channels * 2  # TODO: make this tunable.

    def forward(self, *features, filtered_hsi):
        feature_list = features[-(self.encoder_depth-1):][::-1]
        p_last = self.p_last(feature_list[0])
        ps = [p_last]
        for idx, layer in enumerate(self.p_list):
            p_last = layer(p_last, feature_list[idx+1])
            ps.append(p_last)

        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, ps)]
        x = self.merge(feature_pyramid)
        x = self.merge_hsi([x, filtered_hsi])
        x = self.dropout(x)

        return x
