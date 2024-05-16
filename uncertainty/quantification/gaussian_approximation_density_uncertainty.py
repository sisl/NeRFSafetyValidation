import numpy as np
import torch
from scipy.optimize import minimize


class GaussianApproximationDensityUncertainty:
    def __init__(self, c, d, r):
        """
        Initialize the GaussianApproximationDensityUncertainty class.

        Parameters:
        c (torch.Tensor): The color values.
        d (torch.Tensor): The density values.
        r (torch.Tensor): The rendered color.
        """
        self.c = c
        self.d = d
        self.r = r

        # clamp d vals for numerical stability & remove any nans
        self.d = torch.clamp(self.d, min=-1e8, max=1e8)
        self.d[torch.isnan(self.d)] = 0

        # reshape d vals
        self.d = self.d.view(self.c.shape[0], self.c.shape[1], -1)


    def objective(self, params):
        """
        The objective function for the Maximum Likelihood Estimation (MLE).

        Parameters:
        params (list): A list containing the mean and standard deviation of the volume density.

        Returns:
        float: The value of the objective function.
        """
        mu_d, sigma_d = params
        result = torch.log(torch.sum(self.c**2 * self.d**2 * sigma_d**2)) + (self.r - torch.sum(self.c * mu_d * self.d))**2 / torch.sum(self.c**2 * sigma_d**2 * self.d**2)
        return result.item()

    def optimize(self):
        """
        The optimization to find the parameters that minimize the objective function.

        Returns:
        tuple: The optimized mean and standard deviation of the volume density.
        """
        initial_guess = [torch.mean(self.d).item(), torch.std(self.d).item()]

        # perform the optimization
        result = minimize(self.objective, initial_guess)

        # extract optimized parameters
        mu_d_opt, sigma_d_opt = result.x

        return mu_d_opt, sigma_d_opt
