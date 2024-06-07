# NeRF-Safety-Validation

## Abstract

The validation of safety-critical systems, particularly within the realm of autonomous vehicles, is challenging given the inherent complexity of these systems and the dynamic nature of their operational environments. Addressing this challenge necessitates innovative approaches that can accurately replicate the complexities of real-world scenarios to ensure the robustness of critical components such as motion planning modules. The integration of a surrogate model in the form of a Neural Radiance Field (NeRF) emerges as a compelling strategy to augment the safety validation process. 

By leveraging a NeRF as a surrogate model, exploring potential failure modes and vulnerabilities in safety validation becomes comprehensive. The detailed and realistic scene representations provided by NeRFs align with the objectives of creating a controlled and authentic testing environment, allowing for a more nuanced evaluation of the environment under diverse and challenging conditions. This integration not only enhances the reliability of safety validation but also contributes to the development of more robust and adaptive autonomous systems, ultimately fostering greater confidence in the deployment of autonomous vehicles in real-world scenarios.

In addition, this project introduces uncertainty quantification in the context of NeRFs for safety validation. By employing methods such as the Gaussian Approximation and the Bayesian Laplace Approximation, we can better understand and quantify the uncertainty associated with the surrogate model’s predictions. This is then incorporated into the framework via a reward function within the NeRF simulator. The reward function was designed to continuously sample more likely and certain disturbance vectors, leading to more realistic failure modes. This further enhances the robustness of the safety validation process by providing more reliable and trustworthy results.

---

[NeRF](http://www.matthewtancik.com/nerf) (Neural Radiance Fields) is a method that achieves state-of-the-art results for synthesizing novel views of complex scenes.

[nerf-navigation](https://github.com/mikh3x4/nerf-navigation) is a navigation pipeline using PyTorch and NeRFs.

[Instant-NGP](https://github.com/NVlabs/instant-ngp) is an extension that grants enormous performance boosts in inference and training. This repository for navigation is built off of the PyTorch version of NGP.

[torch-NGP](https://github.com/ashawkey/torch-ngp) is an implementation of Instant-NGP in Pytorch.

## Installation

```bash
git clone https://github.com/jfrausto7/NeRF-Safety-Validation.git
cd NeRF-Safety-Validation
```

### 1) Install CUDA Toolkit + Drivers
Find more information based on your system on the [official installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

### 2) Install requirements with pip
```bash
pip install -r requirements.txt

# (optional) install the tcnn backbone for GPUs with lower architectures
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### 3) Build extensions
```bash
# install all extension modules
bash scripts/install_ext.sh
```

### 4) File Creation
Create `data`, `paths`, `cached`, and `sim_img_cache` folders in the workspace.

### 5) Set up Dataset
Following the canonical data format for NeRFs, your training data from Blender should look like the following:

```                                                                                                                              
├── model_name                                                                                                  
│   ├── test      #Contains test images      
│   │   └── r_0.png           
│   │   └── ...                                                                                                    
│   ├── train                                                                                  
│   ├── val  
│   └── transforms_test.json  
│   └── transforms_train.json
│   └── transforms_val.json
```
To start, we recommend using the Stonehenge scene used in the nerf-nav paper. 
**The training data (stonehenge), pre-trained model (stone_nerf), and Blender mesh (stonehenge.blend) can be found [here](https://drive.google.com/drive/folders/104v_ehsK8joFHpPFZv_x31wjt-FUOe_Y?usp=sharing)**.

#### Custom Datasets

For getting started with a custom scene/dataset, we recommend using the [BlenderNeRF](https://github.com/maximeraafat/BlenderNeRF) add-on. Alternatively, you should be able to run:

```bash
python scripts/colmap2nerf.py --images ./path/to/images --run_colmap
```
You can also record a video rotating around your scene and run:

```bash
python scripts/colmap2nerf.py --video ./path/to/video.mp4 --run_colmap
```

### 6) Set up Blender

Make sure to download [the latest version of Blender](https://www.blender.org/download/). We use Blender as our simulation environment. **Ensure that the command ```blender``` in terminal pulls up a Blender instance.**

**Note: Make sure there is a Camera object in the scene you use.**

### 7) Create Collision Map & SDF

To compute distances and actually determine failure modes, be sure to edit the range parameters in ```createCollisionMap.py``` and run it within Blender on your scene. Then, run ```createSDF.py``` using the collision map it generates. This will create an SDF saved as ```sdf.npy``` which you will need to place in ```validation/utils``` before running the validation script.

## Usage

Make sure to first configure the settings for your validation job in `envConfig.json`. The following settings we configure for safety validation are:

* "simulator" - Simulator to use in validation ('NerfSimulator' or 'BlenderSimulator')
* "stress_test" - Stress test to use in validation ('Monte Carlo' or 'Cross Entropy Method')
* "uq_method" - Uncertainty quantification method to use in safety-masked reward function ('Gaussian Approximation' or 'Bayesian Laplace Approximation')
* "n_simulations" - Number of simulations to run in saftey validation
* "x_range" - minimum and maximum values that the planner can take on the x-axis (e.g. '[-1.15, 0.8]')
* "y_range" - minimum and maximum values that the planner can take on the y-axis
* "z_range" - minimum and maximum values that the planner can take on the z-axis
* "steps" - number of steps to take in trajectory
  
### Training a NeRF

```
python main_nerf.py data/nerf_synthetic/{data_name} --workspace {model_name_nerf} -O --bound {X} --scale 1.0 --dt_gamma 0
```

### Safety Validation (w/ online UQ)
```
python validate.py data/nerf_synthetic/{data_name} --workspace {model_name_nerf} -O --bound {X} --scale 1.0 --dt_gamma 0
```

### Uncertainty Quantification (offline)
```
python uncertain.py data/nerf_synthetic/{data_name} --workspace {model_name_nerf} -O --bound {X} --scale 1.0 --dt_gamma 0
```

### Tested environments
* Ubuntu 20 with torch 1.13.1 & CUDA 11.6 on a GTX 1070 Ti.
* Ubuntu 22 with torch 1.13.1 & CUDA 12.2 on a GTX Titan X.
