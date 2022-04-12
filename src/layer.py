import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, inputs):
        return self.conv_block(inputs)


class DownSample(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_step = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, inputs):
        return self.down_step(inputs)


class UpSample(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample_step = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 2, stride=1),
        )
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, inputs, residual_inputs):
        inputs = self.upsample_step(inputs)
        inputs = self.concat(residual_inputs, inputs)
        inputs = self.conv_block(inputs)

        return inputs

    @staticmethod
    def concat(a, b):
        h_diff = a.shape[2] - b.shape[2]
        w_diff = a.shape[3] - b.shape[3]

        b = F.pad(b, [h_diff // 2, w_diff - w_diff // 2,
                      h_diff // 2, h_diff - h_diff // 2])

        return torch.cat([a, b], dim=1)


class OutConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_path = nn.Conv2d(in_channels, out_channels, 1, stride=1)

    def forward(self, inputs):
        return self.out_path(inputs)
