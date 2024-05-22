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
        self.hessian_approximator = HessianApproximator(self.negative_log_posterior_hessian_wrapper)
        self.lr = lr
        self.X = None
        self.y = None

    def log_prior(self, theta):
        if isinstance(theta, np.ndarray):
            theta = torch.from_numpy(theta)
        return -0.5 * torch.sum((theta - self.prior_mean)**2 / self.prior_std**2)

    def log_likelihood(self, theta, X, y):
        self.set_sigma_net_params(theta)
        y_pred = self.model.density(X)['sigma'].view(-1, 1)
        #print(y_pred)
        return -0.5 * torch.sum((y - y_pred)**2)

    def log_posterior(self, theta, X, y):
        return self.log_prior(theta) + self.log_likelihood(theta, X, y)

    def negative_log_posterior(self, theta, X, y):
        nlp = self.log_posterior(theta, X, y)
        if isinstance(nlp, float):
            nlp = torch.tensor([nlp], dtype=torch.float32, requires_grad=True)
        return -nlp

    def grad_negative_log_posterior(self, theta, X, y):
        epsilon = 1e-5
        grad = np.zeros_like(theta)
        for i in range(len(theta)):
            theta_epsilon = theta.copy()
            theta_epsilon[i] += epsilon
            grad[i] = (self.negative_log_posterior(theta_epsilon, X, y) - self.negative_log_posterior(theta, X, y)) / epsilon
            #print(theta_epsilon, grad[i])
        return grad
    
    def hessian_negative_log_posterior(self, theta, X, y):
        epsilon = 1e-5
        hessian = np.zeros((len(theta), len(theta)))
        for i in range(len(theta)):
            for j in range(len(theta)):
                theta_epsilon = theta.copy()
                theta_epsilon[i] += epsilon
                theta_epsilon[j] += epsilon
                hessian[i, j] = (self.negative_log_posterior(theta_epsilon, X, y) 
                                 - self.negative_log_posterior(theta, X, y)) / (epsilon**2)
        return hessian

    def fit(self, X, y):
        theta_init = np.concatenate([param.detach().cpu().numpy().ravel() for param in self.model.sigma_net.parameters()])
        res = minimize(self.negative_log_posterior, theta_init, args=(X, y), jac=self.grad_negative_log_posterior)
        self.set_sigma_net_params(res.x)
        self.posterior_mean = res.x
        res_tensor = torch.from_numpy(res.x)
        # self.X = X
        # self.y = y
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        # self.posterior_cov = np.linalg.inv(self.hessian_negative_log_posterior(res.x, X, y))
        self.posterior_cov = np.linalg.inv(self.hessian_approximator.compute(res_tensor))
        print('REACHED BEYOND POSTERIOR COV')
        return self
    
    def negative_log_posterior_hessian_wrapper(self, xt):
        return self.negative_log_posterior(xt, self.X, self.y)

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
            if isinstance(updated, np.ndarray):
                new_vals = torch.from_numpy(updated[start:end]).view(param.shape)
            else:  # Assuming it's already a PyTorch tensor
                new_vals = updated[start:end].view(param.shape)
            param.data.copy_(new_vals)
            start = end