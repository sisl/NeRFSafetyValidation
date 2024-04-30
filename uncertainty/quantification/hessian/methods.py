import numpy as np
import torch
from torch.optim import LBFGS
from sklearn.linear_model import LinearRegression, Ridge


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

def regression_gradient(theta, func, perturbations=200, delta=1e-6):
    """
    Estimates the gradient from the results of random perturbations from a point theta using linear regression.

    Parameters:
    theta (numpy.ndarray): The point at which to estimate the gradient.
    func (function): The function whose gradient is to be estimated.
    perturbations (int, optional): The number of random perturbations to generate.
    delta (float, optional): The scale of the random perturbations.

    Returns:
    hessian: the approximated Hessian matrix.
    """

    n = len(theta)

    # perturbation matrix and utility change vector
    delta_theta = np.zeros((perturbations, n))
    delta_u = np.zeros(perturbations)

    # generate random perturbations and calculate utility change
    for i in range(perturbations):
        delta_theta[i] = delta * np.random.randn(n)
        delta_u[i] = func(torch.from_numpy(theta.detach().numpy() + delta_theta[i])).sum().item() - func(theta).sum().item()

    # estimate hessian w/ multivariate regression
    X = np.hstack([delta_theta, 0.5*np.outer(delta_theta, delta_theta).reshape(perturbations, -1)])
    model = LinearRegression().fit(X, delta_u)

    hessian_elements = model.coef_[n:]
    hessian = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            # elements of the hessian are symmetric, so we only need to compute half of them
            index = int(n*i - i*(i-1)/2 + j)
            hessian[i, j] = hessian[j, i] = hessian_elements[index]

    return torch.from_numpy(hessian)

def regression_gradient_regularized(theta, func, perturbations=200, delta=1e-6, alpha=0.1):
    """
    Estimates the gradient from the results of random perturbations from a point theta using linear regression with regularization.

    Parameters:
    theta (numpy.ndarray): The point at which to estimate the gradient.
    func (function): The function whose gradient is to be estimated.
    perturbations (int, optional): The number of random perturbations to generate.
    delta (float, optional): The scale of the random perturbations.
    alpha (float, optional): The regularization strength.

    Returns:
    hessian: the approximated Hessian matrix.
    """

    n = len(theta)

    # perturbation matrix and utility change vector
    delta_theta = np.zeros((perturbations, n))
    delta_u = np.zeros(perturbations)

    # generate random perturbations and calculate utility change
    for i in range(perturbations):
        delta_theta[i] = delta * np.random.randn(n)
        delta_u[i] = func(torch.from_numpy(theta.detach().numpy() + delta_theta[i])).sum().item() - func(theta).sum().item()

    # estimate hessian w/ multivariate regression with regularization
    X = np.hstack([delta_theta, 0.5*np.outer(delta_theta, delta_theta).reshape(perturbations, -1)])
    model = Ridge(alpha=alpha).fit(X, delta_u)

    hessian_elements = model.coef_[n:]
    hessian = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            # elements of the hessian are symmetric, so we only need to compute half of them
            index = int(n*i - i*(i-1)/2 + j)
            hessian[i, j] = hessian[j, i] = hessian_elements[index]

    return torch.from_numpy(hessian)

def levenberg_marquardt(x0, func, lmbda=0.01, max_iter=100):
    """
    Levenberg-Marquardt method for approximating Hessian.

    Parameters:
    x0 (numpy.ndarray): Initial guess for the solution.
    func (function): The function to minimize.
    lmbda (float, optional): Initial damping factor.
    max_iter (int, optional): Maximum number of iterations.

    Returns:
    hessian: the approximated Hessian matrix.
    """
    x = x0.clone().detach().requires_grad_(True)
    for _ in range(max_iter):
        y = func(x)
        g = torch.autograd.grad(y, x, create_graph=True)[0]
        try:
            dx = torch.linalg.solve(torch.outer(g, g) + lmbda * torch.eye(len(x)), -g)
            hessian = torch.outer(g, g).detach()  # Approximate Hessian
        except RuntimeError:
            lmbda *= 10
            continue
        if torch.allclose(dx, torch.zeros_like(dx)):
            break
        x = x + dx
        if func(x) < func(x0):
            lmbda /= 10
        else:
            lmbda *= 10
    return hessian