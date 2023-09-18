import torch.nn as nn
import torch
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, numClass):
        super(FPN, self).__init__()
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Lateral layers [C]
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        # Smooth layers
        # ps0
        self.smooth1_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # ps1
        self.smooth1_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth2_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth3_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth4_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        # ps2 is concatenation
        # Classify layers
        self.smooth = nn.Conv2d(128 * 4, 128 * 4, kernel_size=3, stride=1, padding=1)
        self.classify = nn.Conv2d(128 * 4, numClass, kernel_size=3, stride=1, padding=1)

    def _concatenate(self, p5, p4, p3, p2):
        _, _, H, W = p2.size()
        p5 = F.interpolate(p5, size=(H, W), **self._up_kwargs)
        p4 = F.interpolate(p4, size=(H, W), **self._up_kwargs)
        p3 = F.interpolate(p3, size=(H, W), **self._up_kwargs)
        return torch.cat([p5, p4, p3, p2], dim=1)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.interpolate(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), **self._up_kwargs) + y

    def first_forward(self, features_4x, features_8x, features_16x, features_32x):
        p5 = self.toplayer(features_32x)
        p4 = self._upsample_add(p5, self.latlayer1(features_16x))
        p3 = self._upsample_add(p4, self.latlayer2(features_8x))
        p2 = self._upsample_add(p3, self.latlayer3(features_4x))
        ps0 = [p5, p4, p3, p2]

        return ps0

    def second_forward(self, ps0):
        # Smooth
        p5, p4, p3, p2 = ps0
        p5 = self.smooth1_1(p5)
        p4 = self.smooth2_1(p4)
        p3 = self.smooth3_1(p3)
        p2 = self.smooth4_1(p2)
        ps1 = [p5, p4, p3, p2]

        return ps1

    def third_forward(self, ps1):
        # Smooth
        p5, p4, p3, p2 = ps1
        p5 = self.smooth1_2(p5)
        p4 = self.smooth2_2(p4)
        p3 = self.smooth3_2(p3)
        p2 = self.smooth4_2(p2)
        ps2 = [p5, p4, p3, p2]

        return ps2

    def final_forward(self, ps2):
        # Classify
        # use ps2_ext
        p5, p4, p3, p2 = ps2
        ps3 = self._concatenate(p5, p4, p3, p2)
        ps3 = self.smooth(ps3)
        output = self.classify(ps3)

        return output

    def forward(self, features_4x, features_8x, features_16x, features_32x):
        ps0 = self.first_forward(features_4x, features_8x, features_16x, features_32x)
        ps1 = self.second_forward(ps0)
        ps2 = self.third_forward(ps1)
        output = self.final_forward(ps2)

        return output
