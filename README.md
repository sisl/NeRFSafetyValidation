# NeRF-Safety-Validation

## Abstract

The validation of safety-critical systems, particularly within the realm of autonomous vehicles, is challenging given the inherent complexity of these systems and the dynamic nature of their operational environments. Addressing this challenge necessitates innovative approaches that can accurately replicate the complexities of real-world scenarios to ensure the robustness of critical components such as motion planning modules. The integration of a surrogate model in the form of a Neural Radiance Field (NeRF) emerges as a compelling strategy to augment the safety validation process. 

By leveraging a NeRF as a surrogate model, exploring potential failure modes and vulnerabilities in safety validation becomes comprehensive. The detailed and realistic scene representations provided by NeRFs align with the objectives of creating a controlled and authentic testing environment, allowing for a more nuanced evaluation of the environment under diverse and challenging conditions. This integration not only enhances the reliability of safety validation but also contributes to the development of more robust and adaptive autonomous systems, ultimately fostering greater confidence in the deployment of autonomous vehicles in real-world scenarios.

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

For getting started with a custom scene/dataset, we recommend using the [BlenderNeRF](https://github.com/maximeraafat/BlenderNeRF) add-on to capture images from your Blender scene, saving those images in your workspace, and then running:

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

## Usage

Make sure to first configure the settings for your validation job in `envConfig.json`.

**TODO: ADD DETAILS ABOUT CONFIG**

### Training a NeRF

```
python main_nerf.py data/nerf_synthetic/{data_name} --workspace {model_name_nerf} -O --bound {X} --scale 1.0 --dt_gamma 0
```

### Safety Validation
```
python3 validate.py data/nerf_synthetic/{data_name} --workspace {model_name_nerf} -O --bound {X} --scale 1.0 --dt_gamma 0
```

### Tested environments
* Ubuntu 20 with torch 1.13.1 & CUDA 11.6 on a GTX 1070 Ti.
* Ubuntu 22 with torch 1.13.1 & CUDA 12.2 on a GTX Titan X.
