###################################################################################################
# Title: Split dataset images
# Description: Randomly split images into train and test folders
# 
# Jure Rebernik magistrska naloga
###################################################################################################

import os
import random

def dataset_split(input_folder, output_train_folder, output_test_folder, split_factor):
    # create a list of all jpg images found in the input folder
    jpg_images = [filename for filename in os.listdir(input_folder) if filename.endswith('.jpg')]

    # randomly split the list of jpg images into two lists
    # Split factor: 1 => all images in train folder, 0 => all images in test folder
    random_seed = 123
    random.seed(random_seed)
    random.shuffle(jpg_images)
    num_jpg = len(jpg_images)
    split_index = int(split_factor * num_jpg)
    images_train = jpg_images[:split_index]
    images_test = jpg_images[split_index:]

    for filename in images_train:
        cmd = 'cp {} {}'.format(os.path.join(input_folder, filename), os.path.join(output_train_folder, filename))
        os.system(cmd)

    for filename in images_test:
        cmd = 'cp {} {}'.format(os.path.join(input_folder, filename), os.path.join(output_test_folder, filename))
        os.system(cmd)

if __name__ == '__main__':
    input_folder = 'dataset/images_resized'
    output_train_folder = 'dataset/images/train'
    output_test_folder = 'dataset/images/test'
    split_factor = 0.85

    dataset_split(input_folder, output_train_folder, output_test_folder, split_factor)
