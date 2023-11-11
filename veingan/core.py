import logging
import os.path
from typing import Any, AnyStr, Dict, List

import random
import torch
import torchvision.utils as vutils
import numpy as np

from veingan.dataloader.image import (
    SingleFingerVeinDataset,
    DualFingerVeinDataset,
    FV1C_TRANSFORM,
    FV3C_TRANSFORM,

    AnomalyImageDataset,
    EVALUATE_TRANSFORM
)
from veingan.model.cnn import vgg_extract_features
from veingan.model.svm import OneSVM
from veingan.model.gan import dcgan_train, cyclegan_train
from veingan.model.vae import vae_train, vae_generate_latent_space_out
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


def generate_method_vae(data_dir: AnyStr, target_dir: AnyStr, configuration: AnyStr):
    """
    Generate Images using VAE (Variational Auto Encoder) as Method

    :param data_dir: Path to Dataset
    :param target_dir: Path to Target
    :param configuration: Configuration to pass model for generation
    :return:
    """

    CONFIGURATION = {
        'vae_128+cpu': {
            'image_size': 128,
            'batch_size': 64,
            'input_dim': 128 * 128,
            'h1_dim': 4096,
            'h2_dim': 1024,
            'h3_dim': 512,
            'latent_dim': 256,
            'epoch': 100,
            'ngpu': 0
        },
        'vae_128+gpu': {
            'image_size': 128,
            'batch_size': 64,
            'input_dim': 128 * 128,
            'h1_dim': 4096,
            'h2_dim': 1024,
            'h3_dim': 512,
            'latent_dim': 256,
            'epoch': 100,
            'ngpu': 1
        }
    }
    CONFIGURATION['default'] = CONFIGURATION['vae_128+cpu']
    if configuration not in CONFIGURATION.keys():
        raise ValueError(f'Configuration {configuration} for VAE does not exist. Please check the documentation.')
    CC = CONFIGURATION[configuration]

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and CC['ngpu'] > 0) else "cpu")
    dataloader = SingleFingerVeinDataset.load_from_dir(data_dir=data_dir, transform=FV1C_TRANSFORM).to_dataloader(
        batch_size=CC['batch_size'],
    )
    vae_model = vae_train(dataloader, CC, device)
    vae_generate_latent_space_out(vae_model, target_dir, device, scale=1.0)
    return


def generate_method_gan(data_dir: AnyStr, target_dir: AnyStr, configuration: AnyStr):
    """
    Generate Images using GAN (Generative Adversarial Network) as Method

    :param data_dir: Path to Dataset
    :param target_dir: Path to Target
    :param configuration: Configuration to pass model for generation
    :return:
    """

    # Set random seed for reproducibility
    manualSeed = 42
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)  # Needed for reproducible results

    CONFIGURATION = {
        "gan128_64+cpu": {
            'ngpu': 0,
            'worker': 2,
            'nc': 3,
            'nz': 100,
            'ngf': 128,
            'ndf': 128,
            'batch_size': 64,
            'epoch': 20,
            'lr_G': 1e-5,
            'lr_D': 1e-5,
            'beta1': 0.5
        },
        "gan128_64+gpu": {
            'ngpu': 1,
            'worker': 2,
            'nc': 3,
            'nz': 100,
            'ngf': 128,
            'ndf': 128,
            'batch_size': 64,
            'epoch': 20,
            'lr_G': 1e-5,
            'lr_D': 1e-5,
            'beta1': 0.5
        },
        "gan128_64+full": {
            'ngpu': 1,
            'worker': 2,
            'nc': 3,
            'nz': 100,
            'ngf': 128,
            'ndf': 128,
            'batch_size': 64,
            'epoch': 50,
            'lr_G': 1e-5,
            'lr_D': 1e-5,
            'beta1': 0.5
        },
    }
    CONFIGURATION['default'] = CONFIGURATION['gan128_64+cpu']
    if configuration not in CONFIGURATION.keys():
        raise ValueError(f'Configuration {configuration} for GAN does not exist. Please check the documentation.')
    CC = CONFIGURATION[configuration]

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and CC['ngpu'] > 0) else "cpu")
    dataloader = SingleFingerVeinDataset.load_from_dir(data_dir=data_dir, transform=FV3C_TRANSFORM).to_dataloader(
        batch_size=CC['batch_size'],
        num_workers=CC['worker']
    )
    generated_images = dcgan_train(dataloader, CC, device)

    for i, gen_img in enumerate(generated_images[-1][-CC['batch_size']:]):
        vutils.save_image(gen_img, f'{target_dir}/gan_{i}.png')
        logging.info(f'Saved: {target_dir}/gan_{i}.png')

    return


def generate_method_cyclegan(data_dir: AnyStr, target_dir: AnyStr, configuration: Dict):
    """
    Generate Images using CycleGAN Method

    :param data_dir: Path to Dataset
    :param target_dir: Path to Target
    :param configuration: Configuration to pass model for generation
    :return:
    """

    # Set random seed for reproducibility
    manualSeed = 42
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    CONFIGURATION = {
        "cyclegan128_1+cpu": {
            'ngpu': 0,
            'worker': 2,
            'nc': 3,
            'nr': 9,
            'batch_size': 1,
            'epoch': 20,
            'lr_G': 1e-5,
            'lr_D': 1e-5,
            'beta1': 0.5,
            'lambda_cycle': 10.0,
            'lambda_identity': 1.0
        },
        "cyclegan128_1+gpu": {
            'ngpu': 1,
            'worker': 2,
            'nc': 3,
            'nr': 9,
            'batch_size': 1,
            'epoch': 20,
            'lr_G': 1e-5,
            'lr_D': 1e-5,
            'beta1': 0.5,
            'lambda_cycle': 10.0,
            'lambda_identity': 1.0
        },
        "cyclegan128_1+full": {
            'ngpu': 1,
            'worker': 2,
            'nc': 3,
            'nr': 9,
            'batch_size': 1,
            'epoch': 50,
            'lr_G': 1e-5,
            'lr_D': 1e-5,
            'beta1': 0.5,
            'lambda_cycle': 10.0,
            'lambda_identity': 1.0
        },
    }
    CONFIGURATION['default'] = CONFIGURATION['cyclegan128_1+cpu']
    if configuration not in CONFIGURATION.keys():
        raise ValueError(f'Configuration {configuration} for CycleGAN does not exist. Please check the documentation.')
    CC = CONFIGURATION[configuration]

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and CC['ngpu'] > 0) else "cpu")

    x_dir, y_dir = data_dir.split(';')
    dataloader = DualFingerVeinDataset.load_from_dir(
        data_dir=x_dir, data_dir_p=y_dir, transform=FV3C_TRANSFORM
    ).to_dataloader(
        batch_size=CC['batch_size'],
        num_workers=CC['worker']
    )

    cyclegan_train(dataloader, target_dir, CC, device)
    return


def evaluate_method_osvm_vgg(data_dir: AnyStr, configuration: AnyStr):
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
