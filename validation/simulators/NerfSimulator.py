import pathlib
import shutil
import gym
import torch

from nav import (Estimator, Agent, Planner, vec_to_rot_matrix, rot_matrix_to_vec)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NerfSimulator(gym.Env):
    """Class template for safety validation."""

    def __init__(self, density_fn):
        super(NerfSimulator, self).__init__()

        # TODO: define action and observation space as in example:
        # self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(N_OBSERVATIONS,), dtype=np.float32)
        self.action_space = None
        self.observation_space = None
        self.start_state = self.action_space[0]
        self.start_state = self.action_space[-1]
        self.planner_cfg - {}   # TODO: populate using and env config json
        self.density_fn = density_fn
        self.traj = None
        self.basefolder


    def step(self, action):
        """
        Run one timestep of the environment's dynamics.
        Returns:
            observation (object): agent's observation of the current environment.
            reward (float) : amount of reward returned after previous action.
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        pass

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
        basefolder = pathlib.Path("paths") / pathlib.Path(self.planner_cfg['exp_name'])
        if basefolder.exists():
            print(basefolder, "already exists!")
            shutil.rmtree(basefolder)
        basefolder.mkdir()
        (basefolder / "init_poses").mkdir()
        (basefolder / "init_costs").mkdir()
        (basefolder / "replan_poses").mkdir()
        (basefolder / "replan_costs").mkdir()
        (basefolder / "estimator_data").mkdir()
        self.basefolder = basefolder
        print("created", basefolder)