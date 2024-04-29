import numpy as np
import torch
from scipy.optimize import fmin_bfgs
import numpy.linalg as lin
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


def linesearch_secant(f, d, x):
    epsilon=10**(-5)
    max = 500
    alpha_curr=0
    alpha=10**-5
    y,grad=f(x)
    dphi_zero=np.dot(np.array(grad).T,d)

    dphi_curr=dphi_zero
    i=0

    while np.abs(dphi_curr)>epsilon*np.abs(dphi_zero):
        alpha_old=alpha_curr
        alpha_curr=alpha
        dphi_old=dphi_curr
        y,grad=f(x+alpha_curr*d)
        dphi_curr=np.dot(np.array(grad).T,d)
        alpha=(dphi_curr*alpha_old-dphi_old*alpha_curr)/(dphi_curr-dphi_old)
        i += 1
        if (i >= max) and (np.abs(dphi_curr)>epsilon*np.abs(dphi_zero)):
            print('Line search terminating with number of iterations:')
            print(i)
            print(alpha)
            break
        
    return alpha

def bfgs(x, func):
    """
    Approximate the Hessian using the BFGS method (scipy optimizer).

    Parameters:
    x: the point at which to compute the Hessian.
    func (function): The function whose gradient is to be estimated.


    Returns:
    hessian: the approximated Hessian matrix.
    """
    hessian = np.eye(len(x))
    tol = 1e-20
    y,grad = func(x)
    dist=2*tol
    epsilon = tol
    iter=0

    while lin.norm(grad)>1e-6:
        value,grad=func(x)
        print (grad)
        p=np.dot(-hessian,grad)
        lam = linesearch_secant(func,p,x)
        iter += 1
        xt = x
        x = x + lam*p
        s = lam*p
        dist=lin.norm(s)
        newvalue,newgrad=func(x)
        y = np.array(newgrad)-grad
        rho=1/np.dot(y.T,s)
        s = s.reshape(2,1)
        y = y.reshape(2,1)
        tmp1 = np.eye(2)-rho*np.dot(s,y.T)
        tmp2 = np.eye(2)-rho*np.dot(y,s.T)
        tmp3 = rho*np.dot(s,s.T)
        hessian= np.dot(np.dot(tmp1,hessian),tmp2) + tmp3
        
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