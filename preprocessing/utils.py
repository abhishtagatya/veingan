import cv2
import numpy as np
from skimage.morphology import skeletonize


def invert(img):
    return 255 - img


def binary_thresholder(img, threshold=127):
    _, img_bin = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return img_bin


def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)


def histogram_equalization(img):
    return cv2.equalizeHist(img)


def gaussian_blur(img, ksize, sigma):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def gabor_filter(ksize, sigma, theta, lambd, gamma, psi):
    kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi)
    kern /= np.max(np.abs(kern))
    return kern


def apply_filter(image, filter):
    filtered_img = cv2.filter2D(image, cv2.CV_8UC3, filter)
    return filtered_img


def remove_lines(original_image, gabor_filtered_image, threshold):
    # Threshold the Gabor-filtered image to identify lines
    _, binary_lines = cv2.threshold(
        gabor_filtered_image, threshold, 255, cv2.THRESH_BINARY)

    # Subtract the lines from the original image
    result_image = cv2.subtract(original_image, binary_lines)

    return result_image


def increase_contrast(img, alpha=1.0, beta=0):
    """
    Increase the contrast of an image. 
    `alpha` is the contrast control (1.0-3.0)
    `beta` is the brightness control (0-100)
    """
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted


def thin_lines(image, iterations):
    # Convert the image to binary (assuming lines are white on a black background)
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Define a kernel for erosion
    kernel = np.ones((3, 3), np.uint8)

    # Apply erosion iteratively
    for _ in range(iterations):
        binary_image = cv2.erode(binary_image, kernel, iterations=1)

    return binary_image


def skeletonize_image(binary_image):
    skeleton = skeletonize(binary_image).astype(np.uint8)*255
    return skeleton
