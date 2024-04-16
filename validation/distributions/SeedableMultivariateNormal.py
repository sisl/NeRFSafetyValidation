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
    
    def compute_best_solution(self, simulator):
        best_objective_value = 999999
        simulator.reset()
        for stepNumber in range(12):
            best_solutionMean = self.means[stepNumber]
            best_solutionCov = self.covs[stepNumber]

            dist = torch.distributions.MultivariateNormal(best_solutionMean, best_solutionCov)
            noise = dist.sample()
            print(f"Step {stepNumber} with noise: {noise}")
            isCollision, collisionVal, currentPos = simulator.step(noise)
            best_objective_value = min(best_objective_value, collisionVal)
            print(f"Collision: {isCollision}, Collision Value: {collisionVal}, Current Position: {currentPos}")

            if isCollision: break

        return best_solutionMean, best_solutionCov, best_objective_value
