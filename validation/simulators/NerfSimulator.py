import gym
import torch

from nav import (Estimator, Agent, Planner, vec_to_rot_matrix, rot_matrix_to_vec)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NerfSimulator(gym.Env):
    """Class template for safety validation."""

    def __init__(self):
        super(NerfSimulator, self).__init__()

        # TODO: define action and observation space as in example:
        # self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(N_OBSERVATIONS,), dtype=np.float32)
        self.action_space = None
        self.observation_space = None

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
        pass