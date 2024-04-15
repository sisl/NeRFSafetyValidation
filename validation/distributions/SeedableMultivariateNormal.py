import torch

class SeedableMultivariateNormal:
    def __init__(self, means, covs, noise_seed=None):
        self.means = means
        self.covs = covs
        self.noise_seed = noise_seed
        self.distributions = [torch.distributions.MultivariateNormal(mean, cov) for mean, cov in zip(means, covs)]

    def sample(self):
        if self.noise_seed is not None:
            torch.manual_seed(self.noise_seed.initial_seed())
        return [dist.sample() for dist in self.distributions]

    def log_prob(self, x):
        return torch.stack([dist.log_prob(xi) for dist, xi in zip(self.distributions, x)])
