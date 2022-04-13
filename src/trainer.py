from model import UNet
from dataset import FashionDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


def train():
    transforms = A.Compose([
        A.PadIfNeeded(min_height=512, min_width=512),
        A.RandomCrop(512, 512),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ToTensorV2(),
    ])

    dataset = FashionDataset(
        "/media/numce5/32A802712946E7C2/datasets/CV/segmentation/clothing-co-parsing",
        img_size=512,
        transforms=transforms,
    )
    train_data, val_data = random_split(dataset, [800, 204])
    train_loader = DataLoader(train_data, batch_size=6, num_workers=10, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=6, num_workers=10, shuffle=False)

    n_classes = 4
    model = UNet(3, n_classes).to('cuda')

    trainer = pl.Trainer(gpus=1)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()
