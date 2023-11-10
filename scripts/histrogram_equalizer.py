import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)


def histogram_equalization(img):
    return cv2.equalizeHist(img)


if __name__ == '__main__':

    # create argument parser
    parser = argparse.ArgumentParser(
        description='Histogram equalization of input image')
    parser.add_argument('input_path', type=str, help='path to input image')
    parser.add_argument('output_path', type=str,
                        help='path to save output image')
    parser.add_argument('--method', type=str, default='clahe',
                        help='histogram equalization method')
    parser.add_argument('--clip_limit', type=float, default=2.0,
                        help='clip limit for CLAHE method')
    parser.add_argument('--grid_size', type=int, nargs=2, default=(8, 8),
                        help='tile grid size for CLAHE method')
    parser.add_argument('--show', action='store_true',
                        help='show comparison between original and equalized image')

    # parse arguments
    args = parser.parse_args()

    # read input image
    img = cv2.imread(args.input_path, cv2.IMREAD_GRAYSCALE)

    if args.method == 'clahe':
        clip_limit = args.clip_limit
        img_eq = clahe(img, clip_limit=clip_limit,
                       tile_grid_size=args.grid_size)
    elif args.method == 'histogram_equalization':
        img_eq = histogram_equalization(img)

    out_img = img_eq

    # create directory if not exists
    output_directory = args.output_path[:args.output_path.rfind('/')]
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # save output image
    cv2.imwrite(args.output_path, out_img)

    if args.show:
        # show comparison in one big window
        img_show = cv2.hconcat([img, out_img])
        cv2.imshow('Comparison', img_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
