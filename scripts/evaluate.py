#!/usr/bin/python3

import argparse
import logging

from veingan.core import evaluate_method_osvm_vgg
from veingan.core import evaluate_method_entropy
from veingan.core import evaluate_method_laplace
from veingan.core import evaluate_method_inception
from veingan.core import evaluate_visualize_snapshot

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Images from VeinGAN")
    parser.add_argument("method", help="Method of Evaluation", type=str)
    parser.add_argument("target", help="Target Directory for Evaluation", type=str)
    parser.add_argument("--configuration", help="Configuration for Selected Method")
    parser.add_argument("--verbose", help="Verbosity of Output")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    configuration = args.configuration or 'default'

    if args.method == 'osvm+vgg':
        evaluate_method_osvm_vgg(data_dir=args.target, configuration=configuration)

    if args.method == 'entropy':
        evaluate_method_entropy(data_dir=args.target, configuration=configuration)

    if args.method == 'laplace':
        evaluate_method_laplace(data_dir=args.target, configuration=configuration)

    if args.method == 'inception':
        evaluate_method_inception(data_dir=args.target, configuration=configuration)

    if args.method == 'snapshot':
        evaluate_visualize_snapshot(data_dir=args.target, configuration=configuration)
