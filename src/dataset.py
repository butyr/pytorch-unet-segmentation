import os
from torch.utils.data import Dataset
import scipy.io
import torch
import numpy as np
from torchvision.transforms import Pad
import cv2
from PIL import Image


class FashionDataset(Dataset):
    def __init__(
            self,
            root_dir,
            img_size,
    ):
        self.img_size = img_size
        self.img_dir = os.path.join(root_dir, 'photos')
        self.label_dir = os.path.join(root_dir, 'annotations/pixel-level')

        self.label_files = os.listdir(self.label_dir)

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        label_file = self.label_files[idx]
        label = scipy.io.loadmat(os.path.join(self.label_dir, label_file))['groundtruth']

        img_path = os.path.join(self.img_dir, f"{label_file.split('.')[0]}.jpg")
        image = cv2.imread(img_path)

        label = self.resize(label, self.img_size, image.shape[1:])
        image = self.resize(image, self.img_size, image.shape[1:])

        label = torch.tensor(label, dtype=torch.long)
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1)

        return image, label

    @staticmethod
    def resize(img, img_size, img_shape):
        img = Image.fromarray(img)
        width, height = img_shape

        if width > height:
            padding = Pad((0, 0, 0, width-height))
        else:
            padding = Pad((0, 0, height-width, 0))

        padded_img = padding(img)
        resized_img = cv2.resize(np.array(padded_img), (img_size, img_size), interpolation=cv2.INTER_AREA)

        return resized_img
