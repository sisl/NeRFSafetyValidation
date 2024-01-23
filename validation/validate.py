import numpy as np
import torch
import argparse
from nav.math_utils import vec_to_rot_matrix
from nerf.provider import NeRFDataset
from nerf.utils import PSNRMeter, Trainer, get_rays
from validation.simulators.NerfSimulator import NerfSimulator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####################### MAIN LOOP ##########################################
def validate(simulator):
    simulator.reset()
    simulator.step()
    return

####################### END OF MAIN LOOP ##########################################

if __name__ == "__main__":

    opt = argparse.parse_args()

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.empty_cache()

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = False
        opt.preload = False

    if opt.ff:
        opt.fp16 = False
        assert opt.bg_radius <= 0, "background model is not implemented for --ff"
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        opt.fp16 = False
        assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork

    # TODO: figure out how to seed everything

    model = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
    )

    model.eval()
    metrics = [PSNRMeter(),]
    criterion = torch.nn.MSELoss(reduction='none')
    trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)
    dataset = NeRFDataset(opt, device=device, type='test')        #Importing dataset in order to get the same camera intrinsics as training


    ### ESTIMATOR CONFIGS
    dil_iter = 3        # Number of times to dilate mask around features in observed image
    kernel_size = 5     # Kernel of dilation 
    batch_size = 1024   # How many rays to sample in dilated mask
    lrate_relative_pose_estimation = 1e-3       # State estimator learning rate
    N_iter = 300        # Number of times to perform gradient descent in state estimator

    #Remark: We don't have a measurement noise covariance, or rather we just set it to identity since it's not clear
    #what a covariance on a random batch of pixels should be. 
    sig0 = 1*np.eye(12)     # Initial state covariance
    Q = 1*np.eye(12)        # Process noise covariance

    ### AGENT CONFIGS

    # Extent of the agent body, centered at origin.
    # low_x, high_x
    # low_y, high_y
    # low_z, high_z
    body_lims = np.array([
        [-0.05, 0.05],
        [-0.05, 0.05],
        [-0.02, 0.02]
    ])

    # Discretizations of sample points in x,y,z direction
    body_nbins = [10, 10, 5]

    mass = 1.           # mass of drone
    g = 10.             # gravitational constant
    I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]   # inertia tensor
    path = 'sim_img_cache/'     # Directory where pose and images are exchanged
    blend_file = 'stonehenge.blend'     # Blend file of your scene

    ### PLANNER CONFIGS
    # X, Y, Z
    #STONEHENGE
    start_pos = [0.39, -0.67, 0.2]      # Starting position [x,y,z]
    end_pos = [-0.4, 0.55, 0.16]        # Goal position
    
    # start_pos = [-0.09999999999999926,
    #             -0.8000000000010297,
    #             0.0999999999999695]
    # end_pos = [0.10000000000000231,
    #             0.4999999999996554,
    #             0.09999999999986946]

    # Rotation vector
    start_R = [0., 0., 0.0]     # Starting orientation (Euler angles)
    end_R = [0., 0., 0.0]       # Goal orientation

    # Angular and linear velocities
    init_rates = torch.zeros(3) # All rates

    T_final = 2.                # Final time of simulation
    steps = 20                  # Number of time steps to run simulation

    planner_lr = 0.001          # Learning rate when learning a plan
    epochs_init = 2500          # Num. Gradient descent steps to perform during initial plan
    fade_out_epoch = 0
    fade_out_sharpness = 10
    epochs_update = 250         # Num. grad descent steps to perform when replanning

    ### MPC CONFIGS
    mpc_noise_mean = [0., 0., 0., 0, 0, 0, 0, 0, 0, 0, 0, 0]    # Mean of process noise [positions, lin. vel, angles, ang. rates]
    mpc_noise_std = [2e-2, 2e-2, 2e-2, 1e-2, 1e-2, 1e-2, 2e-2, 2e-2, 2e-2, 1e-2, 1e-2, 1e-2]    # standard dev. of noise

    ### Integration
    start_pos = torch.tensor(start_pos).float()
    end_pos = torch.tensor(end_pos).float()

    # Change rotation vector to rotation matrix 3x3
    start_R = vec_to_rot_matrix( torch.tensor(start_R))
    end_R = vec_to_rot_matrix(torch.tensor(end_R))

    # Convert 12 dimensional to 18 dimensional vec
    start_state = torch.cat( [start_pos, init_rates, start_R.reshape(-1), init_rates], dim=0 )
    end_state   = torch.cat( [end_pos,   init_rates, end_R.reshape(-1), init_rates], dim=0 )


    ### Trajectory optimization config ###
    #Store configs in dictionary
    planner_cfg = {
    "T_final": T_final,
    "steps": steps,
    "lr": planner_lr,
    "epochs_init": epochs_init,
    "fade_out_epoch": fade_out_epoch,
    "fade_out_sharpness": fade_out_sharpness,
    "epochs_update": epochs_update,
    'start_state': start_state.to(device),
    'end_state': end_state.to(device),
    'exp_name': opt.workspace,                  # Experiment name
    'I': torch.tensor(I).float().to(device),
    'g': g,
    'mass': mass,
    'body': body_lims,
    'nbins': body_nbins
    }

    agent_cfg = {
    'dt': T_final/steps,        # Duration of each time step
    'mass': mass,
    'g': g,
    'I': torch.tensor(I).float().to(device)
    }

    camera_cfg = {
    'half_res': False,      # Half resolution
    'white_bg': True,       # White background
    'path': path,           # Directory where pose and images are stored
    'res_x': 800,           # x resolution (BEFORE HALF RES IS APPLIED!)
    'res_y': 800,           # y resolution
    'trans': True,          # Boolean    (Transparency)
    'mode': 'RGBA'          # Can be RGB-Alpha, or just RGB
    }

    blender_cfg = {
    'blend_path': blend_file,
    'script_path': 'viz_func.py'        # Path to Blender script
    }

    filter_cfg = {
    'dil_iter': dil_iter,
    'batch_size': batch_size,
    'kernel_size': kernel_size,
    'lrate': lrate_relative_pose_estimation,
    'N_iter': N_iter,
    'sig0': torch.tensor(sig0).float().to(device),
    'Q': torch.tensor(Q).float().to(device),
    'render_viz': True,
    'show_rate': [20, 100]
    }

    extra_cfg = {
    'mpc_noise_std': torch.tensor(mpc_noise_std),
    'mpc_noise_mean': torch.tensor(mpc_noise_mean)
    }

    ### NeRF Configs ###
    # Querying the density (for the planner)
    #In NeRF training, the camera is pointed along positive z axis, whereas Blender assumes -z, hence we need to rotate the pose
    rot = torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]], device=device, dtype=torch.float32)

    # Rendering from the NeRF functions
    render_fn = lambda rays_o, rays_d: model.render(rays_o, rays_d, staged=True, bg_color=1., perturb=False, **vars(opt))
    get_rays_fn = lambda pose: get_rays(pose, dataset.intrinsics, dataset.H, dataset.W)

    # Grabs density from NeRF Neural Network
    density_fn = lambda x: model.density(x.reshape((-1, 3)) @ rot)['sigma'].reshape(x.shape[:-1])

    simulator = NerfSimulator(start_state, end_state, agent_cfg, planner_cfg, camera_cfg, filter_cfg, extra_cfg, get_rays_fn, render_fn, density_fn)
  
    # Main loop
    validate(simulator)
    
    end_text = 'End of validation'
    print(f'{end_text:.^20}')
