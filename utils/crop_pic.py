
from pathlib import Path
from PIL import Image
import os


# current_dir = os.getcwd()
# parent_dir = os.path.dirname(current_dir)
# input_folder = os.path.join('data', 'raw_img')
# input_path = os.path.join(parent_dir, input_folder)
# output_folder = os.path.join('data', 'img')
# output_path = os.path.join(parent_dir, output_folder)

crop_coords = (200, 90, 1500, 990)


def crop_and_save_image(input_image_path, output_dir, crop_coords):
    """
    Crops an image at specified coordinates and saves the cropped image.
    """
    # Open the image
    with Image.open(input_image_path) as image:
        image = image.convert('RGB')
        # Crop the image
        cropped_image = image.crop(crop_coords)

        # Save the cropped image
        cropped_image.save(output_dir, format='TIFF')


def copy_directory_structure(input_dir, output_dir):
    """
    Copies the directory structure from the input directory to the output directory.
    """
    for root, dirs, files in os.walk(input_dir):
        # Construct the corresponding subdirectory in the output directory
        output_subdir = os.path.join(output_dir, os.path.relpath(root, input_dir))
        os.makedirs(output_subdir, exist_ok=True)


def data_prep(input_path, output_path):
    copy_directory_structure(input_path, output_path)
    for root, dirs, files in os.walk(input_path):
        for file_name in files:
            if file_name.endswith('.tif'):
                # Construct the input and output file paths
                input_file_path = os.path.join(root, file_name)
                output_subdir = os.path.join(output_path, os.path.relpath(root, input_path))
                output_file_path = os.path.join(output_subdir, file_name)
                # Crop and save the image
                crop_and_save_image(input_file_path, output_file_path, crop_coords)
    print('Done!')


if __name__ == '__main__':

    input_path = 'W:\\project deep counting giardia\\04.19.2023_giardia_x10_antibiotic_test\\CamLabLite files'
    output_path = 'W:\\project deep counting giardia\\04.19.2023_giardia_x10_antibiotic_test\\corpped'
    data_prep(input_path, output_path)
