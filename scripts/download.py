#!/usr/bin/python3

import argparse

from veingan.core import download_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download Dataset for VeinGAN")
    parser.add_argument("--dataset", help="Dataset Name or Key", type=str, default="default")
    parser.add_argument("--target", help="Target Directory", type=str, default="./veingan-tmp")
    args = parser.parse_args()

    download_dataset(dataset=args.dataset, target_dir=args.target)
