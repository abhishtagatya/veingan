import argparse
import os

from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Binary thresholder for fingervein image')
    parser.add_argument('input_path', type=str,
                        help='path to input image or directory')
    parser.add_argument('output_path', type=str,
                        help='path to save output directory')
    parser.add_argument('--recursive', action='store_true',
                        help='recursive behavior for directory input')

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    recursive = args.recursive

    if not os.path.isdir(input_path) and os.path.isfile(input_path) and not recursive:
        bmp_image = Image.open(input_path)
        bmp_image.save(output_path, format="PNG")

    elif os.path.isdir(input_path) and recursive:

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.bmp'):
                    n_input_path = os.path.join(root, file)

                    output_filename = file[:file.rfind(
                        '.')] + '.png'
                    n_output_path = os.path.join(output_path, output_filename)
                    bmp_image = Image.open(n_input_path)
                    bmp_image.save(n_output_path, format="PNG")
    elif os.path.isdir(input_path) and not recursive:
        print('Please use --recursive option to preprocess all images in a directory')
    elif os.path.isfile(input_path) and recursive:
        print('Please use --recursive option to preprocess all images in a directory')
    else:
        print('Invalid input path')
