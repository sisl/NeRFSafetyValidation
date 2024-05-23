import os
import pathlib
import shutil
import gym
import numpy as np
import torch
from gym.spaces import Box
import matplotlib.image

from nav import (Estimator, Agent, Planner, vec_to_rot_matrix, rot_matrix_to_vec)
from nerf.utils import seed_everything
from uncertain import uncertainty
from validation.utils.fileUtils import cache_poses, restore_poses
from validation.utils.blenderUtils import worldToIndex

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NerfSimulator(gym.Env):
    """Class template for safety validation."""

    def __init__(self, start_state, end_state, agent_cfg, planner_cfg, camera_cfg, filter_cfg, get_rays_fn, render_fn, blender_cfg, density_fn, seed):
        super(NerfSimulator, self).__init__()

        self.action_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)  # 12-dimensional disturbance vector
        self.observation_space = Box(low=0, high=255, shape=(800, 800, 3), dtype=np.uint8)  # RGB image of size (800, 800)
        self.planner_cfg = planner_cfg
        self.start_state = start_state
        self.end_state = end_state
        self.density_fn = density_fn
        self.camera_cfg = camera_cfg
        self.filter_cfg = filter_cfg
        self.blender_cfg = blender_cfg
        self.get_rays_fn = get_rays_fn
        self.render_fn = render_fn

        # Change start state from 18-vector (with rotation as a rotation matrix) to 12 vector (with rotation as a rotation vector)
        agent_cfg['x0'] = torch.cat([start_state[:6], rot_matrix_to_vec(start_state[6:15].reshape((3, 3))), start_state[15:]], dim=-1).cuda()
        agent_cfg['dt'] = self.planner_cfg['T_final'] / self.planner_cfg['steps']
        agent_cfg['I'] = torch.tensor(agent_cfg['I']).float().to(device)
        self.agent_cfg = agent_cfg
        true_start_state = torch.cat([start_state[:6], rot_matrix_to_vec(start_state[6:15].reshape((3, 3))), start_state[15:]], dim=-1).cuda()
        self.true_start_state = true_start_state
        self.true_states = true_start_state.cpu().detach().numpy()
        self.current_state = None
        self.dynamics = None
        self.filter = None
        self.traj = None
        self.steps = 0
        self.iter = 0

        # collision grid parameters
        self.GRANULARITY = 40
        self.START_X = -1.2
        self.END_x = 0.6
        self.START_Y = -1.2
        self.END_Y = 1.2
        self.START_Z = -0.22
        self.END_Z = 1.2
        self.sdf = np.load("validation/utils/sdf.npy")
        self.seed = seed


    def step(self, disturbance, num_interpolated_points=4):
        """
        Run one timestep of the environment's dynamics. The agent performs the action recommended by the planner,
        subject to the provided disturbance. The function also checks for collisions and updates the state estimate.

        Args:
            disturbance (np.array): The disturbance vector, a 12-dimensional vector representing the disturbance.
            num_interpolated_points (int, optional): The number of interpolated points for linear interpolation on states. Defaults to 4.

        Returns:
            collided (bool): Whether the drone collided during the step.
            collisionVal (float): The value at the collision (sdf).
            current_state[:3] (np.array): The position during collision.
        """
        try:
            # In MPC style, take the next action recommended from the planner
            action = self.traj.get_next_action().clone().detach()

            # Have the agent perform the recommended action, subject to noise. true_pose, true_state are here
            # for simulation purposes in order to benchmark performance. They are the true state of the agent
            # subjected to noise. gt_img is the observation.
            true_pose, true_state, gt_img = self.dynamics.step(action, noise=disturbance)
            self.current_state = true_state
            self.true_states = np.vstack((self.true_states, true_state))
            true_pose = torch.from_numpy(true_pose)
            true_pose = true_pose.to(device)

            # linear interpolation on states
            x = np.arange(self.true_states.shape[0])
            xnew = np.linspace(x.min(), x.max(), self.true_states.shape[0] * num_interpolated_points)
            true_states_interpolated = np.empty((xnew.shape[0], self.true_states.shape[1]))
            for i in range(self.true_states.shape[1]):
                true_states_interpolated[:, i] = np.interp(xnew, x, self.true_states[:, i])
            
            with torch.no_grad():
                print(f"Calling nerf render with pose {true_pose}")
                nerf_image = self.filter.render_from_pose(true_pose)
                nerf_image = torch.squeeze(nerf_image).cpu().detach().numpy()
                nerf_image_reshaped = nerf_image.reshape((800, 800, -1))
                nerf_image_reshaped *= 255
                nerf_image_reshaped = nerf_image_reshaped.astype(np.uint8)
            # convert to torch object

            # calculate uncertainty
            _, sigma = uncertainty("Gaussian Approximation", rendered_output=self.filter.render_for_uncertainty(true_pose))
            
            print("saving image files")
            gt_img_tuple = gt_img.cpu().detach().numpy()
            matplotlib.image.imsave("./sim_img_cache/blenderRender.png", gt_img_tuple)
            matplotlib.image.imsave("./sim_img_cache/NeRFRender.png", nerf_image_reshaped)

            # Given the planner's recommended action and the observation, perform state estimation. true_pose
            # is here only to benchmark performance. 
            true_pose = true_pose.cpu().detach().numpy()
            state_est = self.filter.estimate_state(nerf_image_reshaped, true_pose, action)

            #state estimate is 12-vector. Transform to 18-vector
            state_est = torch.cat([state_est[:6], vec_to_rot_matrix(state_est[6:9]).reshape(-1), state_est[9:]], dim=-1)

            # Let the planner know where the agent is estimated to be
            self.traj.update_state(state_est)

            # Replan from the state estimate
            self.traj.learn_update(self.iter)            

            collisionVal = 9999

            # check for collisions
            for current_state in true_states_interpolated[-num_interpolated_points:]:
                try:
                    x = worldToIndex(current_state[0], self.START_X, self.GRANULARITY)
                    y = worldToIndex(current_state[1], self.START_Y, self.GRANULARITY)
                    z = worldToIndex(current_state[2], self.START_Z, self.GRANULARITY)
                    
                    collisionVal = self.sdf[x, y, z]
                    collided = collisionVal < (1 / self.GRANULARITY) # if we are within 1 grid cell of the surface, we have collided
                except IndexError:
                    print(f"We are out of bounds with current state {current_state}")
                    collided = False

                if collided:
                    print(f"Drone collided in state {current_state}")
                    return collided, collisionVal, current_state[:3], sigma
                else:
                    print(f"Drone did NOT collide in state {current_state}")

            self.iter += 1

            # return if it collided, the value at the collision (sdf), and the position during collision
            return collided, collisionVal, current_state[:3], sigma
        except KeyboardInterrupt:
            return
        
    def reward(self, likelihood, sigma_d_opt):
        """
        Compute the reward based on the uncertainty of the image and the likelihood of the trajectory.

        Parameters:
        likelihood (float): The likelihood of the trajectory.
        sigma_d_opt (float): The optimized standard deviation of the volume density, used as a measure of uncertainty.

        Returns:
        float: The computed reward.
        """
        penalty_strength = 36.0 # slightly above likely disturbance 

        # reward is directly proportional to the likelihood and decreases with increasing uncertainty
        reward = np.clip((likelihood - penalty_strength * sigma_d_opt), -likelihood * 3, 100)

        return reward

    def reset(self):
        """
        Resets the state of the environment.
        """
        self.basefolder = "paths" / pathlib.Path(self.planner_cfg['exp_name'])
        cache_flag = os.path.exists(self.basefolder / pathlib.Path("init_poses") / "0.json")
        self.clear_workspace()
        seed_everything(self.seed)
        self.iter = 0

        # Reinitialize dynamics
        self.dynamics = Agent(self.agent_cfg, self.camera_cfg, self.blender_cfg)

        #Reinitialize estimator
        self.filter = Estimator(self.filter_cfg, self.dynamics, self.true_start_state, get_rays_fn=self.get_rays_fn, render_fn=self.render_fn)

        # Reinitialize Planner
        traj = Planner(self.start_state, self.end_state, self.planner_cfg, self.density_fn)
        traj.basefolder = self.basefolder
        self.filter.basefolder = self.basefolder

        # Create a coarse trajectory to initialize the planner by using A*. 
        traj.a_star_init()

        # From the A* initialization, perform gradient descent on the flat states of agent to get a trajectory
        # that minimizes collision and control effort.
        if not cache_flag:
            traj.learn_init(0)
            init_poses = "paths" / pathlib.Path(self.planner_cfg['exp_name']) / "init_poses"
            init_costs = "paths" / pathlib.Path(self.planner_cfg['exp_name']) / "init_costs"
            target = "cached" / pathlib.Path(self.planner_cfg['exp_name'])
            cache_poses(init_poses, init_costs, target)
        else:
            cached_poses = "cached" / pathlib.Path(self.planner_cfg['exp_name']) / "poses"
            cached_costs = "cached" / pathlib.Path(self.planner_cfg['exp_name']) / "costs"
            target = "paths" / pathlib.Path(self.planner_cfg['exp_name'])
            restore_poses(cached_poses, cached_costs, target)
        

        self.traj = traj
        self.steps = traj.get_actions().shape[0]


    def clear_workspace(self):
        """
        Clears the workspace directory and sim image cache.
        """
        if self.basefolder.exists():
            print(self.basefolder, "already exists!")
            shutil.rmtree(self.basefolder)
            print(self.basefolder, "has been cleared.")
        self.basefolder.mkdir()
        (self.basefolder / "init_poses").mkdir()
        (self.basefolder / "init_costs").mkdir()
        (self.basefolder / "replan_poses").mkdir()
        (self.basefolder / "replan_costs").mkdir()
        (self.basefolder / "estimator_data").mkdir()
        print("created", self.basefolder)

        sim_img_cache = pathlib.Path(self.agent_cfg["path"])
        if sim_img_cache.exists():
            print(sim_img_cache, "already exists!")
            shutil.rmtree(sim_img_cache)
            print(sim_img_cache, "has been cleared.")
        sim_img_cache.mkdir()
        print("created", sim_img_cache)
