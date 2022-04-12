import torch
from layer import DownSample, UpSample, OutConv, ConvBlock
import pytorch_lightning as pl
import torch.nn.functional as F


class UNet(pl.LightningModule):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.in_conv = ConvBlock(in_channels, 64).to('cuda')
        self.down_path = [DownSample(in_channels, out_channels).to('cuda') for in_channels, out_channels in [
            (64, 128), (128, 256), (256, 512), (512, 1024)
        ]]
        self.up_path = [UpSample(in_channels, out_channels).to('cuda') for in_channels, out_channels in [
            (1024, 512), (512, 256), (256, 128), (128, 64)
        ]]
        self.out = OutConv(64, n_classes).to('cuda')

    def forward(self, x):
        x = self.in_conv(x)

        residual_outputs = [x]
        for down_step in self.down_path:
            x = down_step(x)
            residual_outputs.append(x)

        for up_step, r_x in zip(self.up_path, reversed(residual_outputs[:-1])):
            x = up_step(x, r_x)

        return self.out(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x_hat = self.forward(x)
        loss = F.cross_entropy(x_hat, y)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x_hat = self.forward(x)
        loss = F.cross_entropy(x_hat, y)
        self.log('train_loss', loss)

        return loss
