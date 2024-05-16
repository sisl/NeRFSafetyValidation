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
        result = torch.log(torch.sum(self.c**2 * self.d**2 * sigma_d**2)) + (torch.mean(self.r) - torch.sum(self.c * mu_d * self.d))**2 / torch.sum(self.c**2 * sigma_d**2 * self.d**2)
        return result.item()
    
    def optimize(self):
        """
        This function performs an optimization for each element in the tensor 'self.d'. 
        The objective function is minimized for each element using the element value and 1.0 as the initial guess.
        After performing the optimization for all elements, the mean and standard deviation of all the optimized means 
        and standard deviations are calculated. These represent the mean and standard deviation of the optimized density 
        values for the whole image.

        Returns:
        tuple: The mean and standard deviation of the optimized density values for the whole image, 
           and the optimized mean and standard deviation for each pixel.
        """
        mu_d_opt = np.zeros_like(self.d.cpu().numpy())
        sigma_d_opt = np.zeros_like(self.d.cpu().numpy())

        for i in range(self.c.shape[0]):
            for j in range(self.c.shape[1]):
                for k in range(self.c.shape[1]):
                    initial_guess = [self.d[i, j].item(), 1.0]

                    # perform the optimization for pixel (i, j)
                    result = minimize(self.objective, initial_guess, args=(self.c[i, j, k].item(), self.d[i, j].item(), self.r[i, j, k].item()))

                    # extract optimized parameters
                    mu_d_opt[i, j, k], sigma_d_opt[i, j, k] = result.x

        # params for whole image
        mu_d_image = np.mean(mu_d_opt)
        sigma_d_image = np.std(sigma_d_opt)

        return mu_d_image, sigma_d_image, mu_d_opt, sigma_d_opt
