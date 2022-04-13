import torch
from layer import DownSample, UpSample, OutConv, ConvBlock
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import Pad
import cv2
from PIL import Image


class UNet(pl.LightningModule):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.in_conv = ConvBlock(in_channels, 64).to('cuda')
        self.down_0 = DownSample(64, 128).to('cuda')
        self.down_1 = DownSample(128, 256).to('cuda')
        self.down_2 = DownSample(256, 512).to('cuda')
        self.down_3 = DownSample(512, 1024).to('cuda')
        self.down_path = [self.down_0, self.down_1, self.down_2, self.down_3]

        self.up_0 = UpSample(1024, 512).to('cuda')
        self.up_1 = UpSample(512, 256).to('cuda')
        self.up_2 = UpSample(256, 128).to('cuda')
        self.up_3 = UpSample(128, 64).to('cuda')
        self.up_path = [self.up_0, self.up_1, self.up_2, self.up_3]

        self.out = OutConv(64, n_classes).to('cuda')
        self.criterion = torch.nn.CrossEntropyLoss()

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.999, min_lr=1e-8)

        return {'optimizer': optimizer, 'scheduler': scheduler}

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x_hat = self.forward(x)
        loss = self.criterion(x_hat, y)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x_hat = self.forward(x)
        loss = self.criterion(x_hat, y)
        self.log('val_loss', loss)

        if batch_idx == 0:
            img = torch.argmax(x_hat, dim=1, keepdim=True)
            img = img.cpu().detach().numpy()
            img = img/3
            tensorboard = self.logger.experiment

            for i in range(img.shape[0]):
                cm = plt.get_cmap('Set3')
                img_i = cm(np.squeeze(img[i]))
                img_i = img_i.transpose(2, 0, 1)

                tensorboard.add_image(f"samples_{i}", img_i, global_step=self.global_step)
                tensorboard.add_image(f"inputs_{i}", x[i], global_step=self.global_step)

        return loss

    def predict(self, img):
        resized_image = self.resize(img, 512, img.shape[1:])
        image = np.expand_dims(np.array(resized_image), axis=0)
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(0, 3, 1, 2)

        out = self.forward(image.to('cuda'))
        out = torch.argmax(out, dim=1, keepdim=True)
        out = out.permute(0, 2, 3, 1)
        out = out.cpu().detach().numpy()

        return out

    @staticmethod
    def resize(img, img_size, img_shape):
        img = Image.fromarray(img)
        height, width = img_shape

        if width > height:
            padding = Pad((0, 0, 0, width - height))
        else:
            padding = Pad((0, 0, height - width, 0))

        padded_img = padding(img)
        resized_img = cv2.resize(np.array(padded_img), (img_size, img_size), interpolation=cv2.INTER_AREA)

        return resized_img
