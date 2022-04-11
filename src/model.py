import torch
from src.layer import DownSample, UpSample, OutConv, ConvBlock


class UNet(torch.nn.Module):

    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.in_conv = ConvBlock(in_channels, 64)
        self.down = [DownSample(in_channels, out_channels) for in_channels, out_channels in [
            (64, 128), (128, 256), (256, 512), (512, 1024)
        ]]
        self.up = [UpSample(in_channels, out_channels) for in_channels, out_channels in [
            (1024, 512), (512, 256), (256, 128), (128, 64)
        ]]
        self.out = OutConv(64, n_classes)

    def forward(self, x):
        x = self.in_conv(x)

        residual_outputs = [x]
        for down_step in self.down:
            x = down_step(x)
            residual_outputs.append(x)

        for up_step, r_x in zip(self.up, reversed(residual_outputs[:-1])):
            x = up_step(x, r_x)

        return self.out(x)
