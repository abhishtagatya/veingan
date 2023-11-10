#!/usr/bin/python3

import argparse
import logging

from veingan.core import generate_method_gan

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Images using VeinGAN")
    parser.add_argument("model", help="Generative Model", type=str, default="gan")
    parser.add_argument("dataset", help="Training Dataset Directory", type=str, default="/veingan-tmp")
    parser.add_argument("target", help="Target Directory for Generative Result", type=str, default="/veingan-result")
    parser.add_argument("--configuration", help="Configuration for Selected Method")
    parser.add_argument("--verbose", help="Verbosity of Output")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.model == 'gan':
        configuration = args.configuration or 'gan64_64+cpu'
        generate_method_gan(data_dir=args.dataset, target_dir=args.target, configuration=configuration)
