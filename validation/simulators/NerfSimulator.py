import pathlib
import shutil
import gym
import cv2
import numpy as np
import torch
from gym.spaces import Box

from nav import (Estimator, Agent, Planner, vec_to_rot_matrix, rot_matrix_to_vec)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NerfSimulator(gym.Env):
    """Class template for safety validation."""

    def __init__(self, start_state, end_state, agent_cfg, planner_cfg, camera_cfg, filter_cfg, get_rays_fn, render_fn, blender_cfg, density_fn):
        super(NerfSimulator, self).__init__()

        self.action_space = None # TODO: Define vector here
        self.observation_space = Box(low=0, high=255, shape=(800, 800, 3), dtype=np.uint8)  # RGB image of size (800, 800)
        self.planner_cfg = planner_cfg
        self.start_state = start_state
        self.end_state = end_state
        self.density_fn = density_fn

        # Change start state from 18-vector (with rotation as a rotation matrix) to 12 vector (with rotation as a rotation vector)
        agent_cfg['x0'] = torch.cat([start_state[:6], rot_matrix_to_vec(start_state[6:15].reshape((3, 3))), start_state[15:]], dim=-1).cuda()
        true_start_state = torch.cat([start_state[:6], rot_matrix_to_vec(start_state[6:15].reshape((3, 3))), start_state[15:]], dim=-1).cuda()
        self.true_states = true_start_state.cpu().detach().numpy()
        self.dynamics = Agent(agent_cfg, camera_cfg, blender_cfg)
        self.filter = Estimator(filter_cfg, self.dynamics, true_start_state, get_rays_fn=get_rays_fn, render_fn=render_fn)
        self.traj = None


    def step(self, disturbance):
        """
        Run one timestep of the environment's dynamics.
        Returns:
            observation (object): agent's observation of the current environment.
            reward (float) : amount of reward returned after previous action.
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        try:
            # In MPC style, take the next action recommended from the planner
            # if iter < steps - 5:
            #     action = traj.get_next_action().clone().detach()
            # else:
            #     action = traj.get_actions()[iter - steps + 5, :]
            action = self.traj.get_next_action().clone().detach()

            # Have the agent perform the recommended action, subject to noise. true_pose, true_state are here
            # for simulation purposes in order to benchmark performance. They are the true state of the agent
            # subjected to noise. gt_img is the observation.
            true_pose, true_state, gt_img = self.dynamics.step(action, noise=disturbance)
            self.true_states = np.vstack((self.true_states, true_state))
            true_pose = torch.from_numpy(true_pose)
            true_pose = true_pose.to(device)

            # TODO: check for type error
            with torch.no_grad():
                nerf_image = self.filter.render_from_pose(true_pose)
                nerf_image = torch.squeeze(nerf_image).cpu().detach()
                nerf_image_reshaped = nerf_image.reshape((800, 800, -1))
                nerf_image_reshaped = cv2.normalize(nerf_image_reshaped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # convert to torch object

            # Given the planner's recommended action and the observation, perform state estimation. true_pose
            # is here only to benchmark performance. 
            true_pose = true_pose.cpu().detach().numpy()
            state_est = self.filter.estimate_state(nerf_image_reshaped, true_pose, action)

            # if iter < steps - 5:
            #state estimate is 12-vector. Transform to 18-vector
            state_est = torch.cat([state_est[:6], vec_to_rot_matrix(state_est[6:9]).reshape(-1), state_est[9:]], dim=-1)

            # Let the planner know where the agent is estimated to be
            self.traj.update_state(state_est)

            # Replan from the state estimate
            self.traj.learn_update(iter)
            return
        except KeyboardInterrupt:
            return

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        Returns:
            observation (object): the initial observation.
        """
        self.clear_workspace()

        # Reinitialize Planner
        traj = Planner(self.start_state, self.end_state, self.planner_cfg, self.density_fn)
        traj.basefolder = self.basefolder

        # Create a coarse trajectory to initialize the planner by using A*. 
        traj.a_star_init()

        # From the A* initialization, perform gradient descent on the flat states of agent to get a trajectory
        # that minimizes collision and control effort.
        traj.learn_init()

        self.traj = traj


    def clear_workspace(self):
        """Clears the workspace directory."""
        basefolder = "paths" / pathlib.Path(self.planner_cfg['exp_name'])
        if basefolder.exists():
            print(basefolder, "already exists!")
            if input("Clear it before continuing? [y/N]:").lower() == "y":
                shutil.rmtree(basefolder)
        basefolder.mkdir()
        (basefolder / "init_poses").mkdir()
        (basefolder / "init_costs").mkdir()
        (basefolder / "replan_poses").mkdir()
        (basefolder / "replan_costs").mkdir()
        (basefolder / "estimator_data").mkdir()
        print("created", basefolder)
        self.basefolder = basefolder