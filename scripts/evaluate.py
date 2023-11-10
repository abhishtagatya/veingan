#!/usr/bin/python3

import argparse
import logging

from veingan.core import evaluate_method_osvm_vgg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Images from VeinGAN")
    parser.add_argument("method", help="Method of Evaluation", type=str)
    parser.add_argument("target", help="Target Directory for Evaluation", type=str)
    parser.add_argument("--configuration", help="Configuration for Selected Method")
    parser.add_argument("--verbose", help="Verbosity of Output")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.method == 'osvm+vgg':
        configuration = args.configuration or 'osvm+vgg_full'
        evaluate_method_osvm_vgg(data_dir=args.target, configuration=configuration)
