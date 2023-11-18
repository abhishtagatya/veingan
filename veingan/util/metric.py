from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import inception_v3

from scipy.stats import entropy
from scipy.ndimage import laplace
import numpy as np


def calculate_entropy(dataset: Dataset, n_eval: int = 128) -> List:
    """
    Calculate Entropy of Each Image in a Dataset

    :param dataset: Dataset of Image
    :param n_eval: Number of Image to Evaluate
    :return:
    """

    entropy_dist = []
    for idx in range(len(dataset)):
        cur_img = dataset[idx]
        prob = np.histogram(
            cur_img.flatten(), bins=256, range=(0, 256), density=True
        )
        entropy_dist.append(entropy(prob[0], base=2))

    return entropy_dist


def calculate_laplacian_gradient(dataset: Dataset, n_eval: int = 128) -> List:
    """
    Calculate Laplacian Gradient of Each Image in a Dataset

    :param dataset: Dataset of Image
    :param n_eval: Number of Image to Evaluate
    :return:
    """

    laplacian_dist = []
    for idx in range(len(dataset)):
        cur_img = dataset[idx]
        laplace_gradient = laplace(cur_img)
        sum_abs_gradient = np.sum(np.abs(laplace_gradient))

        laplacian_dist.append(sum_abs_gradient)

    return laplacian_dist


def calculate_inception_score(dataloader: DataLoader, batch_size=32, n_eval=128) -> List:
    """
    Calculates Inception Score with InceptionV3

    :param dataloader: Dataloader of Images
    :param batch_size: Batch Size
    :param num_samples: Num Samples to Evaluate
    :return:
    """

    model = inception_v3(pretrained=True, transform_input=False, aux_logits=True)
    model.eval()

    preds = []
    with torch.no_grad():
        for batch in dataloader:
            batch_images = batch[0].unsqueeze(0)
            batch_preds = torch.nn.functional.softmax(model(batch_images), dim=1)
            preds.append(batch_preds)

            if len(preds) * batch_size >= n_eval:
                break

    preds = torch.cat(preds, dim=0)[:n_eval]
    marginal_probs = torch.mean(preds, dim=0)
    kl_divergence = torch.nn.functional.kl_div(preds.log(), marginal_probs.unsqueeze(0), reduction='batchmean')
    inception_score = torch.exp(kl_divergence.mean())
    return inception_score.item()
