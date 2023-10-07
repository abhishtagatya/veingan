#!/usr/bin/python3

import argparse

from veingan.core import dummy_function

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Images using VeinGAN")
    parser.add_argument("model", help="Generative Model", type=str, default="main")
    parser.add_argument("dataset", help="Training Dataset Directory", type=str, default="/veingan-tmp")
    parser.add_argument("target", help="Target Directory for Generative Result", type=str, default="/veingan-result")
    parser.add_argument("--verbose", help="Verbosity of Output")
    parser.add_argument("--epoch", help="Training Epoch", default=10)

    args = parser.parse_args()

    dummy_function(args.model, args.dataset, args.target, args.verbose, args.verbose)  # Just for Demonstration
