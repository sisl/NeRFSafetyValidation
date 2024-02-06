import pathlib
import shutil
import gym
import numpy as np
import torch
from gym.spaces import Box
import matplotlib.image

from nav import (Estimator, Agent, Planner, vec_to_rot_matrix, rot_matrix_to_vec)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BlenderSimulator(gym.Env):
    """Class template for safety validation."""

    def __init__(self, start_state, end_state, agent_cfg, planner_cfg, camera_cfg, filter_cfg, get_rays_fn, render_fn, blender_cfg, density_fn):
        super(BlenderSimulator, self).__init__()

        self.action_space = None # TODO: Define disturbance vector here
        self.observation_space = Box(low=0, high=255, shape=(800, 800, 3), dtype=np.uint8)  # RGB image of size (800, 800) TODO: change this to the state vector
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
            if self.iter < self.steps - 5:
                action = self.traj.get_next_action().clone().detach()
            else:
                action = self.traj.get_actions()[self.iter - self.steps + 5, :]

            # Have the agent perform the recommended action, subject to noise. true_pose, true_state are here
            # for simulation purposes in order to benchmark performance. They are the true state of the agent
            # subjected to noise. gt_img is the observation.
            true_pose, true_state, gt_img = self.dynamics.step(action, noise=disturbance)
            self.true_states = np.vstack((self.true_states, true_state))

            # Given the planner's recommended action and the observation, perform state estimation. true_pose
            # is here only to benchmark performance. 
            state_est = self.filter.estimate_state(gt_img, true_pose, action)

            #state estimate is 12-vector. Transform to 18-vector
            state_est = torch.cat([state_est[:6], vec_to_rot_matrix(state_est[6:9]).reshape(-1), state_est[9:]], dim=-1)

            # Let the planner know where the agent is estimated to be
            self.traj.update_state(state_est)

            # Replan from the state estimate
            self.traj.learn_update(self.iter)

            self.iter += 1
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
        traj.learn_init()

        self.traj = traj
        self.steps = traj.get_actions().shape[0]


    def clear_workspace(self):
        """Clears the workspace directory."""
        basefolder = "paths" / pathlib.Path(self.planner_cfg['exp_name'])
        if basefolder.exists():
            print(basefolder, "already exists!")
            shutil.rmtree(basefolder)
            print(basefolder, "has been cleared.")
        basefolder.mkdir()
        (basefolder / "init_poses").mkdir()
        (basefolder / "init_costs").mkdir()
        (basefolder / "replan_poses").mkdir()
        (basefolder / "replan_costs").mkdir()
        (basefolder / "estimator_data").mkdir()
        print("created", basefolder)
        self.basefolder = basefolder