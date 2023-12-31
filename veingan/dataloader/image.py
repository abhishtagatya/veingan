import glob
import random

import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def _as_numpy(image):
    return np.array(image)


FV1C_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.CenterCrop((128, 128)),
    transforms.ToTensor(),
])

FV3C_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop((128, 128)),
    transforms.ToTensor(),
])

EVALUATE_TENSOR_1C_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.CenterCrop((128, 128)),
    transforms.ToTensor(),
])

EVALUATE_TENSOR_3C_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop((128, 128)),
    transforms.ToTensor(),
])

EVALUATE_NUMPY_1C_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.CenterCrop((128, 128)),
    _as_numpy
])

EVALUATE_NUMPY_3C_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop((128, 128)),
    _as_numpy
])


class SingleFingerVeinDataset(Dataset):

    def __init__(self, data, transform=None):
        super(SingleFingerVeinDataset, self).__init__()
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    @classmethod
    def load_from_dir(cls, data_dir, transform=None):
        dir_list = glob.glob(data_dir)
        return cls(data=dir_list, transform=transform)

    def to_dataloader(self, batch_size: int = 1, shuffle: bool = True, num_workers: int = 1):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


class DualFingerVeinDataset(Dataset):

    def __init__(self, data, data_p, transform=None):
        super(DualFingerVeinDataset, self).__init__()
        self.data = data
        self.data_p = data_p
        self.transform = transform

        if len(self.data) != len(self.data_p):
            raise ValueError(
                f'Inconsistent length between `data` ({len(self.data)}) and `data_p` ({len(self.data_p)}). '
            )

    def __len__(self):
        return len(self.data)

    @classmethod
    def load_from_dir(cls, data_dir, data_dir_p, transform=None):
        dir_list = glob.glob(data_dir)
        dir_list_p = glob.glob(data_dir_p)
        return cls(data=dir_list, data_p=dir_list_p, transform=transform)

    def to_dataloader(self, batch_size: int = 1, shuffle: bool = True, num_workers: int = 1):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        img_p = Image.open(self.data_p[idx]).convert('RGB')

        if self.transform:
            img = self.transform(img)
            img_p = self.transform(img_p)

        return img, img_p


class EvaluateDataset(Dataset):

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    @classmethod
    def load_from_dir(cls, data_dir, transform=None, shuffle: bool = False):
        dir_list = glob.glob(data_dir)

        if shuffle:
            random.shuffle(dir_list)

        return cls(data=dir_list, transform=transform)

    def to_dataloader(self, batch_size: int = 1, shuffle: bool = True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img
