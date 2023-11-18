import logging
import os

try:
    import kaggle
    KAGGLE_LOCK = False
except OSError:
    KAGGLE_LOCK = True


def download_kaggle_dataset(dataset: str, data_dir: str = '.data/'):
    sub_folder = '/dataset/'
    full_dir = os.getenv('VG_DIR', data_dir) + sub_folder

    if not os.path.exists(full_dir):
        os.makedirs(full_dir)

    if KAGGLE_LOCK:
        logging.error(f'Can\'t fetch dataset from Kaggle. ')
        return

    logging.info(f'Fetching Dataset {dataset} from Kaggle.')
    kaggle.api.dataset_download_files(dataset, full_dir, unzip=True)
    return
