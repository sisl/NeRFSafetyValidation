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
        r (float): The rendered color.
        """
        self.c = c
        self.d = d
        self.r = r

    def objective(self, params):
        """
        The objective function for the Maximum Likelihood Estimation (MLE).

        Parameters:
        params (list): A list containing the mean and standard deviation of the volume density.

        Returns:
        float: The value of the objective function.
        """
        mu_d, sigma_d = params
        return np.log(np.sum(self.c**2 * self.d**2 * sigma_d**2)) + (self.r - np.sum(self.c * mu_d * self.d))**2 / np.sum(self.c**2 * sigma_d**2 * self.d**2)

    def optimize(self):
        """
        The optimization to find the parameters that minimize the objective function.

        Returns:
        tuple: The optimized mean and standard deviation of the volume density.
        """
        # Initial guess
        initial_guess = [torch.mean(self.d).item(), torch.std(self.d).item()]

        # Perform the optimization
        result = minimize(self.objective, initial_guess)

        # Extract the optimized parameters
        mu_d_opt, sigma_d_opt = result.x

        return mu_d_opt, sigma_d_opt
