from model import UNet
from dataset import FashionDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl


def train():
    dataset = FashionDataset(
        "/media/numce5/32A802712946E7C2/datasets/CV/segmentation/clothing-co-parsing",
        img_size=512,
    )
    train_data, val_data = random_split(dataset, [800, 204])
    train_loader = DataLoader(train_data, batch_size=8, num_workers=16)
    val_loader = DataLoader(val_data, batch_size=8, num_workers=16)

    n_classes = 4
    model = UNet(3, n_classes).to('cuda')

    trainer = pl.Trainer(gpus=1)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()
