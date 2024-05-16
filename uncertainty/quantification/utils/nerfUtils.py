import json
import os

def load_camera_params(image_name, dataset_path):
    """
    Load the camera parameters for a given image.

    Parameters:
    image_name (str): The name of the image.
    dataset_path (str): The path to the dataset directory.

    Returns:
    dict: The camera parameters for the image.
    """

    # Remove the file extension from the image name
    image_name = os.path.splitext(image_name)[0]

    # Load the transforms.json file
    with open(os.path.join(dataset_path, 'transforms.json'), 'r') as f:
        transform = json.load(f)

    # Find the camera parameters for the image
    for frame in transform['frames']:
        if frame['file_path'] == image_name:
            return frame['transform_matrix']

    raise ValueError(f"Camera parameters for image {image_name} not found.")