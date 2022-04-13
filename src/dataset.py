import os
from torch.utils.data import Dataset
import scipy.io
import torch
import numpy as np
from torchvision.transforms import Pad
import cv2
from PIL import Image
import torch.nn.functional as F


class FashionDataset(Dataset):
    def __init__(
            self,
            root_dir,
            img_size,
            transforms=None,
    ):
        self.img_size = img_size
        self.img_dir = os.path.join(root_dir, 'photos')
        self.label_dir = os.path.join(root_dir, 'annotations/pixel-level')

        self.label_files = os.listdir(self.label_dir)
        self.transforms = transforms

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        label_file = self.label_files[idx]
        label = scipy.io.loadmat(os.path.join(self.label_dir, label_file))['groundtruth']
        label = self.reduce_classes(label)

        img_path = os.path.join(self.img_dir, f"{label_file.split('.')[0]}.jpg")
        image = cv2.imread(img_path)

        label = self.pad(label, image.shape[:-1])
        image = self.pad(image, image.shape[:-1])

        image, label = np.array(image), np.array(label)
        transformed = self.transforms(image=image, mask=label)
        image, label = transformed['image'], transformed['mask']

        image = torch.as_tensor(np.array(image), dtype=torch.float32)
        label = torch.as_tensor(np.array(label), dtype=torch.int64).reshape(self.img_size, self.img_size)

        return image, label

    @staticmethod
    def reduce_classes(label):
        label = np.where(label == 1, np.ones_like(label) * 3, label)
        label = np.where(label == 2, np.ones_like(label) * 3, label)
        label = np.where(label == 41, np.ones_like(label), label)
        label = np.where(label == 19, np.ones_like(label) * 2, label)
        label = np.where(label > 2, np.ones_like(label) * 3, label)

        return label

    @staticmethod
    def pad(img, img_shape):
        img = Image.fromarray(img)
        height, width = img_shape

        if width > height:
            padding = Pad((0, 0, 0, width-height))
        else:
            padding = Pad((0, 0, height-width, 0))

        padded_img = padding(img)

        return padded_img
