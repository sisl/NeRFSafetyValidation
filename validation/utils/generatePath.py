import json
import random
import numpy as np

def calculate_steps(start_position, end_position, step_size=0.09):  # calculated step size based on stonehenge
    total_distance = np.linalg.norm(np.array(end_position) - np.array(start_position))
    num_steps = round(total_distance / step_size)
    return num_steps

def save_coords(start_position, end_position, steps):
    with open('results/coordinates.json', 'w') as f:
        json.dump({'start_position': start_position, 'end_position': end_position, 'steps': steps}, f)

def load_coords():
    with open('results/coordinates.json', 'r') as f:
        data = json.load(f)
    return data['start_position'], data['end_position'], data['steps']

def generate_path(x_range, y_range, z_range):
    # generate random start and end points within given bounds
    start_position = [random.uniform(low, high) for low, high in [x_range, y_range, z_range]]
    end_position = [random.uniform(low, high) for low, high in [x_range, y_range, z_range]]
    
    # calculate number of steps from start to end
    num_steps = calculate_steps(start_position, end_position)
    
    return start_position, end_position, num_steps
