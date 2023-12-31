import argparse
import os

from preprocessing.utils import *


def isolate_finger(thresh_image):
    # Find contours in the binary image
    contours, _ = cv2.findContours(
        thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Identify the contour corresponding to the central section
    central_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the central section
    mask = np.zeros_like(thresh_image)
    cv2.drawContours(mask, [central_contour], -1, 255, thickness=cv2.FILLED)

    # Keep only the central section using the mask
    result_image = cv2.bitwise_and(thresh_image, mask)

    return result_image


def preprocess_contour(img, clip_limit=16, grid_size=(8, 8), gaussian_blur_ksize=5, gaussian_blur_sigma=1, clip_limit_2=10, grid_size_2=(8, 8), threshold=110):
    img_eq = clahe(img, clip_limit, grid_size)
    gaussian_blur_img = gaussian_blur(
        img_eq, gaussian_blur_ksize, gaussian_blur_sigma)
    inverted_gaussian_img = invert(gaussian_blur_img)
    inverted_gaussian_img_eq = clahe(
        inverted_gaussian_img, clip_limit_2, grid_size_2)
    img_thresh = binary_thresholder(
        inverted_gaussian_img_eq, threshold=threshold)
    isolate_finger_img = isolate_finger(img_thresh)
    skeleton = skeletonize_image(isolate_finger_img)
    return skeleton


def preprocess_grabcut(img, clip_limit=32, grid_size=(4, 4), gaussian_blur_ksize=5, gaussian_blur_sigma=1,
                       clip_limit_2=10, grid_size_2=(12, 12), block_size=63, c=5):
    imgs_grabcut_mask = grabcut_segment_mask(img)
    img_eq = clahe(img, clip_limit, grid_size)
    img_eq2 = clahe(img_eq, clip_limit, grid_size)
    img_eq3 = clahe(img_eq2, clip_limit, grid_size)
    img_eq = img_eq3
    gaussian_blur_img = gaussian_blur(
        img_eq, gaussian_blur_ksize, gaussian_blur_sigma)
    inverted_gaussian_img = invert(gaussian_blur_img)
    inverted_gaussian_img_cl = clahe(
        inverted_gaussian_img, clip_limit_2, grid_size_2)
    inverted_gaussian_img_eq = histogram_equalization(inverted_gaussian_img_cl)
    img_thresh = adaptive_threshold(
        inverted_gaussian_img_eq, block_size=block_size, c=c)
    img_thresh_grabcut_mask = apply_mask(img_thresh, imgs_grabcut_mask)
    ing_thresh_grabcut_mask_cleaned = remove_noise(img_thresh_grabcut_mask)
    img_skeleton = skeletonize_image(ing_thresh_grabcut_mask_cleaned)
    return img_skeleton


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
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        img_preprocessed = preprocess_grabcut(img)
        input_filename = os.path.basename(input_path)
        input_filename, input_file_extension = os.path.splitext(
            input_filename)
        output_filename = input_filename + '_skeleton' + input_file_extension
        output_path = os.path.join(output_path, output_filename)
        cv2.imwrite(output_path, img_preprocessed)
    elif os.path.isdir(input_path) and recursive:

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.bmp'):
                    img = cv2.imread(os.path.join(root, file),
                                     cv2.IMREAD_GRAYSCALE)
                    img_preprocessed = preprocess_v2(img)

                    output_filename = file[:file.rfind(
                        '.')] + '_skeleton' + file[file.rfind('.'):]
                    n_output_path = os.path.join(output_path, output_filename)

                    cv2.imwrite(n_output_path, img_preprocessed)
    elif os.path.isdir(input_path) and not recursive:
        print('Please use --recursive option to preprocess all images in a directory')
    elif os.path.isfile(input_path) and recursive:
        print('Please use --recursive option to preprocess all images in a directory')
    else:
        print('Invalid input path')
