import numpy as np
import torch
from scipy.optimize import fmin_bfgs


def finite_difference(x, func, epsilon):
        """
        Approximate the Hessian using the finite difference method.

        Parameters:
        x: the point at which to compute the Hessian.

        Returns:
        hessian: the approximated Hessian matrix.
        """
        n = x.numel()
        size = x.size()
        hessian = torch.zeros(n, n)
        f_x = func(x)
        grad_x = torch.autograd.grad(f_x, x, create_graph=True)[0]

        for i in range(n):
            x_i = x.clone()
            x_i[i] += epsilon
            f_x_i = func(x_i)
            grad_x_i = torch.autograd.grad(f_x_i, x_i, create_graph=True)[0]
            hessian[i] = (grad_x_i - grad_x) / epsilon

        return hessian.view(*size, *size)

def bfgs(x):
    """
    Approximate the Hessian using the BFGS method (scipy optimizer).

    Parameters:
    x: the point at which to compute the Hessian.

    Returns:
    hessian: the approximated Hessian matrix.
    """
    x = x.detach().numpy()
    func = lambda x: func(torch.tensor(x)).item()
    grad = lambda x: torch.autograd.grad(func(torch.tensor(x)), torch.tensor(x))[0].numpy()
    hessian_inv = fmin_bfgs(func, x, fprime=grad, disp=False, full_output=True)[3]
    hessian = torch.inverse(torch.tensor(hessian_inv, dtype=torch.float))
    return hessian

def regression_gradient(theta, func, perturbations=100, delta=0.01):
    """
    Estimates the gradient from the results of random perturbations from a point theta using linear regression.

    Parameters:
    theta (numpy.ndarray): The point at which to estimate the gradient.
    func (function): The function whose gradient is to be estimated.
    perturbations (int, optional): The number of random perturbations to generate.
    delta (float, optional): The scale of the random perturbations.

    Returns:
    gradient: The estimated gradient of the function at theta.
    """

    n = len(theta)

    # perturbation matrix and utility change vector
    delta_theta = np.zeros((perturbations, n))
    delta_u = np.zeros(perturbations)

    # generate random perturbations and calculate utility change
    for i in range(perturbations):
        delta_theta[i] = delta * np.random.randn(n)
        delta_u[i] = func(theta + delta_theta[i]) - func(theta)

    # estimate gradient w/ linear regression
    gradient = np.linalg.pinv(delta_theta) @ delta_u

    return gradient