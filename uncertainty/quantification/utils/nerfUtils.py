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
    with open(os.path.join(dataset_path, 'transforms_train.json'), 'r') as f:
        transform = json.load(f)

    # find the camera parameters for the image
    for frame in transform['frames']:
        if frame['file_path'] == image_name:
            return frame['transform_matrix']

    raise ValueError(f"Camera parameters for image {image_name} not found.")

def create_heatmap(mu_d_opt, sigma_d_opt):
    hist, xedges, yedges = np.histogram2d(mu_d_opt, sigma_d_opt, bins=50)
    plt.imshow(hist, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.colorbar(label='Count')

    plt.xlabel('mu_d_opt')
    plt.ylabel('sigma_d_opt')
    
    plt.savefig(f'results/uncertainty_heatmap.png')
    plt.show()
    plt.close()