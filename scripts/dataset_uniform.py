import os
import shutil
import sys

from PIL import Image

if len(sys.argv) < 4:
    print("Usage: python dataset_uniform.py <input_directory> <output_directory> <resolution>")
    sys.exit(1)

input_directory = sys.argv[1]
output_directory = sys.argv[2]
resolution = sys.argv[3]

if not os.path.isdir(input_directory):
    print(f"{input_directory} is not a directory")
    sys.exit(1)

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# if resolution not power of 2
resolution = int(resolution)
if not (resolution & (resolution - 1) == 0):
    print(f"{resolution} is not a power of 2")
    sys.exit(1)

for dirs in os.listdir(input_directory):
    dir_name = dirs

    width, height = resolution, resolution

    left_fingers_path = input_directory + "/" + dir_name + "/" + "left" + "/"
    for left_finger in os.listdir(left_fingers_path):
        if left_finger.endswith(".db"):
            continue
        left_finger_img = Image.open(
            left_fingers_path + left_finger)
        left_finger_img = left_finger_img.transpose(Image.FLIP_TOP_BOTTOM)
        left_finger_img = left_finger_img.resize((width, height))
        left_finger_img.save(output_directory + "/" +
                             dir_name + "_leftflipped_"+left_finger)

    right_fingers_path = input_directory + "/" + dir_name + "/" + "right" + "/"
    for right_finger in os.listdir(right_fingers_path):
        if right_finger.endswith(".db"):
            continue
        right_finger_img = Image.open(
            right_fingers_path + right_finger)
        right_finger_img = right_finger_img.resize((width, height))
        right_finger_img.save(output_directory + "/" +
                              dir_name + "_rightflipped_"+right_finger)
