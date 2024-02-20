import torch
from torch.distributions import MultivariateNormal
import numpy as np

from validation.stresstests.CrossEntropyMethod import CrossEntropyMethod

class ToySimulator:
    def __init__(self, collision_threshold):
        # self.position = np.array([0.0, 0.0])
        self.position = torch.tensor([0.0, 0.0])
        self.collision_threshold = collision_threshold

    def reset(self):
        # self.position = np.array([0.0, 0.0])
        self.position = torch.tensor([0.0, 0.0])


    def step(self, noise):
        self.position += noise
        collision_value = -np.linalg.norm(self.position - torch.tensor([5.0, 5.0]))
        is_collision = np.linalg.norm(self.position) > self.collision_threshold
        return is_collision, collision_value, self.position

# f = lambda current_position, goal_position: -np.linalg.norm(current_position - goal_position)
q = [MultivariateNormal(torch.zeros(2), torch.eye(2)*0.25) for _ in range(12)]
p = [MultivariateNormal(torch.zeros(2), torch.eye(2)*0.25) for _ in range(12)]
collision_threshold = 10.0
goal_position = np.array([5.0, 5.0])

noise_mean = torch.zeros(12, 2)
noise_std = torch.ones(12, 2)

# noise_mean and noise_std are 12-d lists of means and standard deviations
# create 12 noise means and 12 noise standard deviations, and add them to the lists
# noise_mean = torch.zeros(12, 2)
# noise_meanAll = []

simulator = ToySimulator(collision_threshold)
cem = CrossEntropyMethod(simulator, q, p, 10, 3, 50, None, None)

means, covs, q, best_solutionMean, best_solutionCov, best_objective_value = cem.optimize()
