import torch

class SeedableMultivariateNormal:
    """
    This class represents a multivariate normal distribution that can be seeded for reproducibility.
    
    Attributes:
        means (Tensor): The means of the multivariate normal distributions.
        covs (Tensor): The covariance matrices of the multivariate normal distributions.
        noise_seed (int, optional): The seed for the random number generator. If provided, it ensures that the samples generated are reproducible.
        distributions (list): A list of MultivariateNormal distributions initialized with the provided means and covariances.
    """
    def __init__(self, means, covs, noise_seed=None):
        self.means = means
        self.covs = covs
        self.noise_seed = noise_seed
        self.distributions = [torch.distributions.MultivariateNormal(means, covs) for _ in range(len(means))]

    def sample(self):
        if self.noise_seed is not None:
            torch.manual_seed(self.noise_seed.initial_seed())
        return [dist.sample() for dist in self.distributions]

    def log_prob(self, x):
        return torch.stack([dist.log_prob(xi) for dist, xi in zip(self.distributions, x)])
