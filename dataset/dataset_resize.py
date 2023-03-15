###################################################################################################
# Title: Resize images in dataset
# Description: Using OpenCV, resize images so that the model learing is faster.
# 
# Jure Rebernik magistrska naloga
###################################################################################################

import cv2
import os

def dataset_resize_images(input_folder, output_folder, output_size):
    # create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # create a list of all jpg images found in the input folder
    jpg_images = [filename for filename in os.listdir(input_folder) if filename.endswith('.jpg')]

    # print the number of jpg images found in the input folder
    num_jpg = len(jpg_images)
    print(f"Found {num_jpg} .jpg images.")
    print('Resizing to {}'.format(output_size))

    # loop through the list of jpg images and resize each one
    for i, filename in enumerate(jpg_images):
        # read the image file
        img = cv2.imread(os.path.join(input_folder, filename))

        # resize the image to 1024x1024
        img_resized = cv2.resize(img, output_size)

        # write the resized image to the output folder
        cv2.imwrite(os.path.join(output_folder, filename), img_resized)

        # print a message for each image that is resized
        print(f"Resized image {i+1} of {num_jpg}: {filename}")

    print("Done resizing images.")

if __name__ == '__main__':

    # specify the input and output folders
    input_folder = "dataset/collected_images/"
    output_folder = "dataset/images_resized/"
    output_size = (1365, 1024)

    dataset_resize_images(input_folder, output_folder, output_size)