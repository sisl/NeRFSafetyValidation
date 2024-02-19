import torch
from torch.distributions import MultivariateNormal
import numpy as np

from validation.stresstests.CrossEntropyMethod import CrossEntropyMethod

class ToySimulator:
    def __init__(self, collision_threshold):
        self.position = np.array([0.0, 0.0])
        self.collision_threshold = collision_threshold

    def reset(self):
        self.position = np.array([0.0, 0.0])

    def step(self, noise):
        self.position += noise
        is_collision = np.linalg.norm(self.position) > self.collision_threshold
        return is_collision, self.position

f = lambda current_position, goal_position: -np.linalg.norm(current_position - goal_position)
q = MultivariateNormal(torch.zeros(2), torch.eye(2))
p = MultivariateNormal(torch.zeros(2), torch.eye(2))
collision_threshold = 10.0
goal_position = np.array([5.0, 5.0])

simulator = ToySimulator(collision_threshold)
cem = CrossEntropyMethod(simulator, f, q, p, 3, 2, 2, None, None)

means, covs, q, best_solution, best_objective_value = cem.optimize()
