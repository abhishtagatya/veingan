from typing import List

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def calculate_entropy(dataset: Dataset) -> List:
    """
    Calculate Entropy of Each Image in a Dataset

    :param dataset: Dataset of Image
    :return:
    """

    entropy_dist = []
    for idx in dataset:
        hist = torch.histc(
            dataset[idx].flatten(),
            bins=dataset[idx].max().item() + 1,
            min=0, max=dataset[idx].max().item()
        )
        probabilities = hist / hist.sum()
        probabilities[probabilities == 0] = 1e-10

        entropy_value = -torch.sum(probabilities * torch.log(probabilities))
        entropy_dist.append(entropy_value.item())

    return entropy_dist
