from copy import deepcopy
import os
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
import argparse
from nav.math_utils import vec_to_rot_matrix
from nerf.provider import NeRFDataset
from nerf.utils import PSNRMeter, Trainer, get_rays, seed_everything
from uncertainty.quantification.bayesian_laplace import BayesianLaplace
from uncertainty.quantification.gaussian_approximation_density_uncertainty import GaussianApproximationDensityUncertainty
from uncertainty.quantification.utils.nerfUtils import create_heatmap, load_camera_params
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H = W = 800

####################### MAIN LOOP ##########################################
def uncertainty(method, path_to_images=None, rendered_output=None, model_to_use=None, lr=None):
    """
    Compute the uncertainty using the specified method.

    Parameters:
    method (str): Name of the uncertainty computation method.
    """
    ac, au = 0, 0
    if method == "Gaussian Approximation":
        print(f"Starting Gaussian Approximation for Uncertainty Quantification of Volume Density")
        results = {"optimized_mu_d": [], "optimized_sigma_d": []}
        varNames = ["optimized_mu_d", "optimized_sigma_d"]
        if path_to_images is not None:
            for i, image_name in enumerate(os.listdir(path_to_images)):
                # OFFLINE METHOD

                # load corresponding camera parameters
                image_name = f'./train/{image_name}'
                cam_param = load_camera_params(image_name, opt.path)
                cam_param = torch.tensor([cam_param])

                # render the image using NeRF
                rays = get_rays_fn(cam_param)
                rays_o = rays["rays_o"].reshape((H, W, -1))
                rays_d = rays["rays_d"].reshape((H, W, -1))

                with torch.no_grad():
                    output = render_fn(rays_o.reshape((1, -1, 3)), rays_d.reshape((1, -1, 3)))

                # extract color/density values
                c = output['rgbs']
                d = output['sigmas']

                # extract rendered color
                r = output['image']            

                # optimize parameters
                gaussian_approximation = GaussianApproximationDensityUncertainty(c, d, r)
                mu_d_opt, sigma_d_opt = gaussian_approximation.optimize()

                # check for absolute certain/uncertain values
                if sigma_d_opt <= 0:
                    ac += 1
                elif sigma_d_opt >= 3:
                    au += 1
                else:
                    results["optimized_mu_d"].append(mu_d_opt)
                    results["optimized_sigma_d"].append(sigma_d_opt)

                print(f"Image #{i} ({image_name}): mu_d_opt = {mu_d_opt}, sigma_d_opt = {sigma_d_opt}")
        else:
            # ONLINE METHOD
            # extract color/density values
            c = rendered_output[0]['rgbs']
            d = rendered_output[0]['sigmas']

            # extract rendered color
            r = rendered_output[0]['image']            

            # optimize parameters
            gaussian_approximation = GaussianApproximationDensityUncertainty(c, d, r)
            mu_d_opt, sigma_d_opt = gaussian_approximation.optimize()

            print(f"mu_d_opt = {mu_d_opt}, sigma_d_opt = {sigma_d_opt}")
            return mu_d_opt, sigma_d_opt
        create_heatmap(results["optimized_mu_d"], results["optimized_sigma_d"])

    elif method == "Bayesian Laplace Approximation":
        print(f"Starting Bayesian Laplace Approximation for Uncertainty Quantification of Volume Density")
        results = {"trace": [], "sdu": []}
        varNames = ["trace", "sdu"]
        if path_to_images is not None:
            model_copy = deepcopy(model)
            theta_copy = np.concatenate([param.detach().cpu().numpy().ravel() for param in model.sigma_net.parameters()])
            for i, image_name in enumerate(os.listdir(path_to_images)):
                # OFFLINE METHOD

                # reset params
                start = 0
                for param in model_copy.sigma_net.parameters():
                    end = start + param.numel()
                    if isinstance(theta_copy, np.ndarray):
                        new_vals = torch.from_numpy(theta_copy[start:end]).view(param.shape)
                    else:  # Assuming it's already a PyTorch tensor
                        new_vals = theta_copy[start:end].view(param.shape)
                    param.data.copy_(new_vals)
                    start = end

                # load corresponding camera parameters
                image_name = f'./train/{image_name}'
                cam_param = load_camera_params(image_name, opt.path)
                cam_param = torch.tensor([cam_param])

                # render the image using NeRF
                rays = get_rays_fn(cam_param)
                rays_o = rays["rays_o"].reshape((H, W, -1))
                rays_d = rays["rays_d"].reshape((H, W, -1))
                X = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2)

                with torch.no_grad():
                    output = render_fn(rays_o.reshape((1, -1, 3)), rays_d.reshape((1, -1, 3)))

                # extract aggregated density values
                d = output['aggregated_density']           

                # initialize BayesianLaplace object
                prior_mean = 0.0
                prior_std = 1.0
                bayesian_laplace = BayesianLaplace(model_copy, prior_mean, prior_std, opt.lr)

                # fit the model
                bayesian_laplace.fit(X, d)

                # get optimized parameters
                pos_mu = bayesian_laplace.get_posterior_mean()
                pos_cov = bayesian_laplace.get_posterior_cov()
                n = pos_cov.shape[0]

                # Trace of the covariance matrix
                trace = np.trace(pos_cov) / n
                
                # Standard deviation of diag elements of covariance matrix
                std_dev_uncertainty = np.sqrt(np.mean(np.diag(pos_cov))) / n

                # frobenius_norm = np.linalg.norm(pos_cov, ord='fro') / n
                # print("FROBENIUS NORM")
                # print(frobenius_norm)

                # check for absolute certain/uncertain values
                # if pos_cov <= 0:
                #     ac += 1
                # elif pos_cov >= 3:
                #     au += 1
                # else:
                # results["pos_mu"].append(pos_mu)
                # results["pos_cov"].append(pos_cov)
                results["trace"].append(trace)
                results["sdu"].append(std_dev_uncertainty)

                # delete tensors and free up GPU memory
                del bayesian_laplace, pos_mu, pos_cov
                torch.cuda.empty_cache()

                print(f"Image #{i} ({image_name}): trace = {trace}, sdu = {std_dev_uncertainty}")
        else:
            # ONLINE METHOD
            # extract aggregated density values
            d = rendered_output[0]['aggregated_density']       

            rays_o = rendered_output[1].reshape((H, W, -1))
            rays_d = rendered_output[2].reshape((H, W, -1))
            X = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2)

            # initialize BayesianLaplace object
            prior_mean = 0.0
            prior_std = 1.0
            bayesian_laplace = BayesianLaplace(model_to_use, prior_mean, prior_std, lr)

            # fit the model
            bayesian_laplace.fit(X, d)

            # get optimized parameters
            pos_mu = bayesian_laplace.get_posterior_mean()
            pos_cov = bayesian_laplace.get_posterior_cov()
            n = pos_cov.shape[0]

            # Trace of the covariance matrix
            trace = np.trace(pos_cov) / n
            
            # Standard deviation of diag elements of covariance matrix
            std_dev_uncertainty = np.sqrt(np.mean(np.diag(pos_cov))) / n

            print(f"trace = {trace}, sdu = {std_dev_uncertainty}")

            # delete tensors and free up GPU memory
            del bayesian_laplace, pos_mu, pos_cov
            torch.cuda.empty_cache()

            return trace, std_dev_uncertainty
        create_heatmap(results["trace"], results["sdu"])
    else:
        print(f"Unrecognized uncertainty quantification method {method}")
        exit()

    # visualize uncertainty
    print(f'Number of absolute certain values: {ac}')
    print(f'Number of absolute uncertain values: {au}')
    for varName in varNames:
        plt.hist(results[varName], bins=50)
        plt.xlabel(f'Uncertainty ({varName})')
        plt.ylabel('Frequency')
        plt.savefig(f'results/uncertainty_{method}_{varName}.png')
        plt.show()
        plt.close()
    return

####################### END OF MAIN LOOP ##########################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=random.randint(0, 99999999))

    ### training options
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    opt = parser.parse_args()

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

    with open('envConfig.json', 'r') as json_file:
        envConfig = json.load(json_file)
    print(f"Reading environment parameters from envConfig.json:\n{envConfig}")

    ### PLANNER CONFIGS
    planner_cfg = envConfig["planner_cfg"]

    seed_everything(opt.seed)

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


    ### ESTIMATOR CONFIGS (ripped directly from JSON)
    #Remark: We don't have a measurement noise covariance, or rather we just set it to identity since it's not clear
    #what a covariance on a random batch of pixels should be. 
    sig0 = 1*np.eye(12)     # Initial state covariance
    Q = 1*np.eye(12)        # Process noise covariance

    ### AGENT CONFIGS
    agent_cfg = envConfig["agent_cfg"]

    # Extent of the agent body, centered at origin.
    # low_x, high_x
    # low_y, high_y
    # low_z, high_z
    body_lims = np.array(agent_cfg["body_lims"])

    # Discretizations of sample points in x,y,z direction
    body_nbins = agent_cfg["body_nbins"]

    mass = agent_cfg["mass"]           # mass of drone
    g = agent_cfg["g"]             # gravitational constant
    I =  agent_cfg["I"]   # inertia tensor
    path = agent_cfg["path"]     # Directory where pose and images are exchanged
    blend_file = agent_cfg["blend_file"]     # Blend file of your scene

    # X, Y, Z
    #STONEHENGE
    start_pos = planner_cfg["start_pos"]      # Starting position [x,y,z]
    end_pos = planner_cfg["end_pos"]        # Goal position
    
    # start_pos = [-0.09999999999999926,
    #             -0.8000000000010297,
    #             0.0999999999999695]
    # end_pos = [0.10000000000000231,
    #             0.4999999999996554,
    #             0.09999999999986946]

    # Rotation vector
    start_R = planner_cfg["start_R"]      # Starting orientation (Euler angles)
    end_R = planner_cfg["end_R"]        # Goal orientation

    # Angular and linear velocities
    init_rates = torch.zeros(3) # All rates

    T_final = planner_cfg["T_final"]       # Final time of simulation
    # steps = planner_cfg["steps"]                   # Number of time steps to run simulation

    planner_lr = planner_cfg["planner_lr"]           # Learning rate when learning a plan
    epochs_init = planner_cfg["epochs_init"]           # Num. Gradient descent steps to perform during initial plan
    fade_out_epoch = planner_cfg["fade_out_epoch"] 
    fade_out_sharpness = planner_cfg["fade_out_sharpness"] 
    epochs_update = planner_cfg["epochs_update"]         # Num. grad descent steps to perform when replanning

    ### MPC CONFIGS
    mpc_cfg = envConfig["mpc_cfg"]
    mpc_noise_mean = mpc_cfg["mpc_noise_mean"]    # Mean of process noise [positions, lin. vel, angles, ang. rates]
    mpc_noise_std = mpc_cfg["mpc_noise_std"]     # standard dev. of noise

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

    camera_cfg = envConfig["camera_cfg"]
    camera_cfg["path"] = path

    blender_cfg = {
    'blend_path': blend_file,
    'script_path': 'viz_func.py'        # Path to Blender script
    }

    filter_cfg = envConfig["estimator_cfg"]
    filter_cfg["sig0"] = torch.tensor(sig0).float().to(device)
    filter_cfg["Q"] = torch.tensor(Q).float().to(device)

    extra_cfg = {
    'mpc_noise_std': torch.tensor(mpc_noise_std),
    'mpc_noise_mean': torch.tensor(mpc_noise_mean)
    }

    simulator_cfg = envConfig["simulator"]
    n_simulations = envConfig["n_simulations"]
    stress_test = envConfig["stress_test"]
    uq_method = envConfig["uq_method"]

    ### NeRF Configs ###
    # Querying the density (for the planner)
    #In NeRF training, the camera is pointed along positive z axis, whereas Blender assumes -z, hence we need to rotate the pose
    rot = torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]], device=device, dtype=torch.float32)

    # Grabs density from NeRF Neural Network
    density_fn = lambda x: model.density(x.reshape((-1, 3)) @ rot)['sigma'].reshape(x.shape[:-1])
    # Rendering from the NeRF functions
    render_fn = lambda rays_o, rays_d: model.render(rays_o, rays_d, staged=True, bg_color=1., perturb=False, **vars(opt))
    get_rays_fn = lambda pose: get_rays(pose, dataset.intrinsics, dataset.H, dataset.W)
  
    uncertainty(uq_method, path_to_images=os.path.join(opt.path, "train"))
  
    end_text = 'End of uncertainty computation'
    print(f'{end_text:.^20}')