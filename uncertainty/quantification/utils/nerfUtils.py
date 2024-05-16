import json
import os

from matplotlib import pyplot as plt
import numpy as np

def load_camera_params(image_name, dataset_path):
    """
    Load the camera parameters for a given image.

    Parameters:
    image_name (str): The name of the image.
    dataset_path (str): The path to the dataset directory.

    Returns:
    dict: The camera parameters for the image.
    """

    # remove the file extension from the image name
    image_name = os.path.splitext(image_name)[0]

    # load the transforms.json file
    with open(os.path.join(dataset_path, 'transforms.json'), 'r') as f:
        transform = json.load(f)

    # find the camera parameters for the image
    for frame in transform['frames']:
        if frame['file_path'] == image_name:
            return frame['transform_matrix']

    raise ValueError(f"Camera parameters for image {image_name} not found.")

def plot_heatmap_on_image(image, image_name, uncertainty):
    # ensure image is normalized
    image = image / np.max(image)

    # ensure uncertainty is normalized
    uncertainty = uncertainty / np.max(uncertainty)

    # display image
    plt.imshow(image)

    # overlay heatmap
    plt.imshow(uncertainty, cmap='jet', alpha=0.5)  # alpha is for transparency

    # add colorbar
    plt.colorbar()
        
    plt.savefig(f'results/uncertainty/{image_name}.png')
    plt.show()
    plt.close()