import logging
import os

import kaggle


def download_kaggle_dataset(dataset: str, data_dir: str = '.data/'):
    sub_folder = '/dataset/'
    full_dir = os.getenv('VG_DIR', data_dir) + sub_folder

    if not os.path.exists(full_dir):
        os.makedirs(full_dir)

    logging.info(f'Fetching Dataset {dataset} from Kaggle.')
    kaggle.api.dataset_download_files(dataset, full_dir, unzip=True)
    return
