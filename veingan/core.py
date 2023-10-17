import logging
from typing import Any, AnyStr, Dict, List

import numpy as np
import pandas as pd

from veingan.dataloader.image import (
    AnomalyImageDataset,
    EVALUATE_TRANSFORM
)
from veingan.model.cnn import vgg_extract_features
from veingan.model.svm import OneSVM
from veingan.util.download import download_kaggle_dataset
from veingan.util.tabulate import create_evaluation_table


def dummy_function(a, *args, **kwargs):
    print(a, args, kwargs)


def download_dataset(dataset: AnyStr, target_dir: AnyStr):
    """
    Download Dataset from Kaggle

    :param dataset: Kaggle Dataset Name or Key
    :param target_dir: Target Directory for Dataset
    :return:
    """

    # List of Recognize Dataset for VeinGAN
    dataset_dict = {
        'kaggle-fv': 'ryeltsin/finger-vein'
    }
    dataset_dict['default'] = dataset_dict['kaggle-fv']

    if dataset in dataset_dict:
        download_kaggle_dataset(dataset_dict[dataset], target_dir)
        return

    download_kaggle_dataset(dataset, target_dir)
    return


def evaluate_method_osvm_vgg(data_dir: AnyStr, configuration: Dict):
    """
    Evaluation Method using OneSVM + VGG16 Novelty Score

    :param data_dir: Path to Dataset
    :param configuration: Configuration to load model for evaluation
    :return:
    """

    # Load Models
    CONFIGURATION = {
        'osvm+vgg_full': './pretrained/osvm/osvm+vgg_scale-poly-0.01_full.sav',
        'osvm+vgg_right': './pretrained/osvm/osvm+vgg_scale-poly-0.01_right.sav',
        'osvm+vgg_left': './pretrained/osvm/osvm+vgg_scale-poly-0.01_left.sav',
    }
    CONFIGURATION['default'] = CONFIGURATION['osvm+vgg_full']
    if configuration not in CONFIGURATION.keys():
        raise ValueError(f'Configuration {configuration} for OSVM+VGG does not exist. Please check the documentation.')

    one_svm = OneSVM.load_pretrained(pretrained_file=CONFIGURATION[configuration])

    # Load Test Loader
    test_dataloader = AnomalyImageDataset.load_from_dir(data_dir=data_dir, transform=EVALUATE_TRANSFORM).to_dataloader()
    ext_features = vgg_extract_features(dataloader=test_dataloader, flatten=True)

    # Evaluate
    ext_predict = one_svm.predict(ext_features)
    ext_scores = one_svm.score_samples(ext_features)

    result_table = create_evaluation_table(data={
        'Mean Score': np.mean(ext_scores),
        'Max. Score': np.max(ext_scores),
        'Min. Score': np.min(ext_scores),
        'Novelty %': f'{(len(ext_predict[ext_predict == 1]) / len(ext_predict)) * 100}%'
    })
    logging.info(f'\n{result_table}')
    return
