import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torch.utils.data import DataLoader

class CustomImageDataset(Dataset):
    def __init__(self, data_dict,image_shape, transform=None, target_transform=None):

        # self.classes = classes_dict
        self.data_dict = data_dict
        self.image_shape = image_shape
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data_dict["numpy target"])

    def __getitem__(self, idx):
        img_path = self.data_dict["image path"][idx]
        image = read_image(img_path)
        image = transforms.Resize(self.image_shape)(image)
        label = np.load(self.data_dict["numpy target"][idx])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
