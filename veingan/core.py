from typing import Any, AnyStr

from veingan.util.download import download_kaggle_dataset


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

