import numpy as np
import torch
from torch.optim import LBFGS
import pdb


def finite_difference(x, func, epsilon):
        """
        Approximate the Hessian using the finite difference method.

        Parameters:
        x: the point at which to compute the Hessian.
        func (function): The function whose gradient is to be estimated.
        epsilon: constant to be added

        Returns:
        hessian: the approximated Hessian matrix.
        """
        n = x.numel()
        size = x.size()
        hessian = torch.zeros(n, n)
        f_x = func(x)
        grad_x = torch.autograd.grad(f_x, x, create_graph=True, allow_unused=True)[0]

        for i in range(n):
            x_i = x.clone().detach()
            x_i.requires_grad_(True)
            x_i_plus_epsilon = x_i.clone()  # create a new tensor to avoid in-place operation
            x_i_plus_epsilon[i] = x_i[i] + epsilon
            f_x_i = func(x_i_plus_epsilon)
            grad_x_i = torch.autograd.grad(f_x_i, x_i_plus_epsilon, create_graph=False, allow_unused=True)[0]
            hessian[i] = (grad_x_i - grad_x) / epsilon

        return hessian.view(*size, *size)

def lbfgs(x, func):
    """
    Approximate the Hessian using the LBFGS method.

    Parameters:
    x: the point at which to compute the Hessian.
    func (function): The function whose gradient is to be estimated.


    Returns:
    hessian: the approximated Hessian matrix.
    """
    # optimizer
    optimizer = LBFGS([x], lr=1)

    def closure():
        optimizer.zero_grad()
        output = func(x)
        output.backward(create_graph=True)
        return output

    optimizer.step(closure)


    grad = x.grad
    hessian = torch.zeros(len(x), len(x))

    for i in range(len(grad)):
        grad[i].backward(retain_graph=True)
        hessian[i] = x.grad
        x.grad.zero_()

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
        delta_u[i] = func(torch.from_numpy(theta.detach().numpy() + delta_theta[i])).sum().item() - func(theta).sum().item()

    # estimate Hessian w/ multivariate regression
    hessian = np.linalg.pinv(delta_theta.T @ delta_theta) @ delta_theta.T @ delta_u

    return torch.from_numpy(hessian).reshape(n, n)