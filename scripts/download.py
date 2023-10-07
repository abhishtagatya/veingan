#!/usr/bin/python3

import argparse

from veingan.core import dummy_function

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download Dataset for VeinGAN")
    parser.add_argument("--target", help="Target Directory", type=str, default="/veingan-tmp")
    args = parser.parse_args()

    dummy_function(args.target)  # Just for Demonstration
