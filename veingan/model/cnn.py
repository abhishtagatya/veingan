from typing import List

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision


class VGG16_Feature(nn.Module):

    def __init__(self):
        super(VGG16_Feature, self).__init__()
        self.features = torchvision.models.vgg16(pretrained=True)  # Load the pre-trained VGG16 model
        self.features = nn.Sequential(*list(self.features.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x


def vgg_extract_features(dataloader: DataLoader, flatten: bool = True):
    ext_features = []

    model = VGG16_Feature()
    model.eval()

    # Feature Extract
    with torch.no_grad():
        for _, images in enumerate(tqdm(dataloader)):
            outputs = model(images)
            ext_features.append(outputs)

    if flatten:
        return np.array(
            [torch.flatten(img).numpy()
             for batch in ext_features
             for img in batch]
        )

    return np.array(ext_features)
