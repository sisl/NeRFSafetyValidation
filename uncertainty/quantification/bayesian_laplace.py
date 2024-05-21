import numpy as np
from scipy.optimize import minimize
import torch

from uncertainty.quantification.hessian.HessianApproximator import HessianApproximator

class BayesianLaplace:
    def __init__(self, model, prior_mean, prior_std, lr):
        """
        Initialize the BayesianLaplace class.

        Parameters:
        model (object): The model to be used.
        prior_mean (float): The mean of the prior distribution.
        prior_std (float): The standard deviation of the prior distribution.
        lr (float): Learning rate for NeRF model.
        """
        
        self.model = model
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.hessian_approximator = HessianApproximator(self.negative_log_posterior)
        self.lr = lr

    def log_prior(self, theta):
        return -0.5 * torch.sum((theta - self.prior_mean)**2 / self.prior_std**2)

    def log_likelihood(self, theta, X, y):
        self.set_sigma_net_params(theta)
        y_pred = self.model.density(X)['sigma'].view(-1, 1)
        #print(y_pred)
        return -0.5 * torch.sum((y - y_pred)**2)

    def log_posterior(self, theta, X, y):
        return self.log_prior(theta) + self.log_likelihood(theta, X, y)

    def negative_log_posterior(self, theta, X, y):
        return -self.log_posterior(theta, X, y)

    def grad_negative_log_posterior(self, theta, X, y):
        epsilon = 1e-5
        grad = np.zeros_like(theta)
        for i in range(len(theta)):
            theta_epsilon = theta.copy()
            theta_epsilon[i] += epsilon
            grad[i] = (self.negative_log_posterior(theta_epsilon, X, y) - self.negative_log_posterior(theta, X, y)) / epsilon
            #print(theta_epsilon, grad[i])
        return grad

    def fit(self, X, y):
        theta_init = np.concatenate([param.detach().cpu().numpy().ravel() for param in self.model.sigma_net.parameters()])
        res = minimize(self.negative_log_posterior, theta_init, args=(X, y), jac=self.grad_negative_log_posterior)
        self.set_sigma_net_params(res.x)
        self.posterior_mean = res.x
        self.posterior_cov = np.linalg.inv(self.hessian_approximator.compute(res.x))
        return self


    def predict(self, X):
        return self.model.forward(X)

    def get_posterior_mean(self):
        return self.posterior_mean

    def get_posterior_cov(self):
        return self.posterior_cov
    
    def set_sigma_net_params(self, updated):
        # set params of sigma_net
        start = 0
        for param in self.model.sigma_net.parameters():
            end = start + param.numel()
            new_vals = torch.from_numpy(updated[start:end]).view(param.shape)
            param.data.copy_(new_vals)
            start = end