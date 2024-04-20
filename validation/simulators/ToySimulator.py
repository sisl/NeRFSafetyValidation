import torch
import numpy as np

from validation.distributions.SeedableMultivariateNormal import SeedableMultivariateNormal
from validation.stresstests.CrossEntropyMethod import CrossEntropyMethod

class ToySimulator:
    def __init__(self, collision_threshold):
        self.position = torch.tensor([0.0, 0.0])
        self.collision_threshold = collision_threshold

    def reset(self):
        self.position = torch.tensor([0.0, 0.0])

    def step(self, noise):
        self.position += noise
        collision_value = -np.linalg.norm(self.position - torch.tensor([5.0, 5.0]))
        is_collision = np.linalg.norm(self.position) > self.collision_threshold
        return is_collision, collision_value, self.position

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
noise_seed = torch.Generator(device=device)
noise_meanQ = [torch.zeros(2)]*2
noise_covQ = torch.stack([torch.eye(2)*0.25]*2)
q = SeedableMultivariateNormal(noise_meanQ, noise_covQ, noise_seed=noise_seed)
p = SeedableMultivariateNormal(noise_meanQ, noise_covQ, noise_seed=noise_seed)
collision_threshold = 10.0
goal_position = np.array([5.0, 5.0])

noise_mean = torch.zeros(12, 2)
noise_std = torch.ones(12, 2)

simulator = ToySimulator(collision_threshold)
cem = CrossEntropyMethod(simulator, q, p, 10, 3, 50, noise_seed, None, None)

means, covs, q, best_solutionMean, best_solutionCov, best_objective_value = cem.optimize()
