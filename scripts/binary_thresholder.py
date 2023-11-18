import argparse
import os

import cv2


def invert(img):
    return 255 - img


def binary_thresholder(img, threshold=127):
    ret, img_bin = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return img_bin


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Binary thresholder for fingervein image')
    parser.add_argument('input_path', type=str, help='path to input image')
    parser.add_argument('output_path', type=str,
                        help='path to save output image')
    parser.add_argument('--threshold', type=int, default=127,
                        help='threshold value for binary thresholding')
    # option to show comparison
    parser.add_argument('--show', action='store_true',
                        help='show comparison between original and binary image')

    args = parser.parse_args()

    img = cv2.imread(args.input_path, cv2.IMREAD_GRAYSCALE)

    img_i = invert(img)

    threshold = args.threshold
    img_bin = binary_thresholder(img_i, threshold=threshold)

    output_directory = args.output_path[:args.output_path.rfind('/')]
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    cv2.imwrite(args.output_path, img_bin)

    if args.show:
        # show comparison in one big window
        img_show = cv2.hconcat([img, img_bin])
        cv2.imshow('Comparison', img_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
