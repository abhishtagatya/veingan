#!/usr/bin/python3

import argparse

from veingan.core import dummy_function

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Images from VeinGAN")
    parser.add_argument("target", help="Target Directory for Evaluation", type=str, default="/veingan-result")
    parser.add_argument("--verbose", help="Verbosity of Output")

    args = parser.parse_args()

    dummy_function(args.target, args.verbose)  # Just for Demonstration
