import shutil
import os

def cache_poses(pose_file_path, cost_file_path, destination_dir):
    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)
    
    # Create the 'cached' directory within the destination directory
    cached_dir = os.path.join(destination_dir, 'cached')
    
    # Create individual directories for each file within the 'cached' directory
    pose_dir = os.path.join(cached_dir, 'poses')
    cost_dir = os.path.join(cached_dir, 'costs')
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(cost_dir, exist_ok=True)
    
    # Copy the files
    shutil.copy(pose_file_path, pose_dir)
    shutil.copy(cost_file_path, cost_dir)
    print("Caching posts & costs!")

def restore_poses(cached_pose_dir, cached_cost_dir, destination_dir):
    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)
    
    # Create the destination directories for each file
    pose_dir = os.path.join(destination_dir, 'init_poses')
    cost_dir = os.path.join(destination_dir, 'init_costs')
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(cost_dir, exist_ok=True)
    
    # Get the list of files in each cached directory
    pose_files = os.listdir(cached_pose_dir)
    cost_files = os.listdir(cached_cost_dir)
    
    # Copy the files back to the respective destination directories
    for file in pose_files:
        shutil.copy(os.path.join(cached_pose_dir, file), pose_dir)
        
    for file in cost_files:
        shutil.copy(os.path.join(cached_cost_dir, file), cost_dir)
    print("Using cached posts & costs!")

