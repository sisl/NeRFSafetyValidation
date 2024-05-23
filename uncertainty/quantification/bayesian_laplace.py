import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from uncertainty.quantification.hessian.HessianApproximator import HessianApproximator

class SpatialDeformationLayer(nn.Module):
    def __init__(self, grid_size):
        super().__init__()
        self.grid_size = grid_size
        self.deformation_grid = nn.Parameter(torch.randn(grid_size, grid_size, grid_size, 3) * 0.1)

    def forward(self, x):
        return F.grid_sample(x.unsqueeze(0), self.deformation_grid.unsqueeze(0)).squeeze(0)
    
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
        self.hessian_approximator = HessianApproximator(self.negative_log_posterior_hessian_wrapper, method='levenberg_marquardt')
        self.lr = lr
        self.X = None
        self.y = None

    def log_prior(self, theta):
        if isinstance(theta, np.ndarray):
            theta = torch.from_numpy(theta)
        return -0.5 * torch.sum((theta - self.prior_mean)**2 / self.prior_std**2)

    def log_likelihood(self, theta, X, y):
        self.set_sigma_net_params(theta)
        y_pred = self.model.density(X)['sigma'].view(1, -1)
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
        theta_init = torch.tensor(theta_init, requires_grad=True).cuda()
        X = torch.tensor(X).cuda()
        y = torch.tensor(y).cuda()

        # Initialize the spatial deformation layer
        deformation_layer = SpatialDeformationLayer(grid_size=16).cuda()

        optimizer = torch.optim.Adam([theta_init, deformation_layer.deformation_grid], lr=self.lr)

        minLoss, minTheta = float('inf'), theta_init
        for _ in range(1000):
            optimizer.zero_grad()
            X_deformed = deformation_layer(X)
            loss = self.negative_log_posterior(theta_init, X_deformed, y)
            loss.backward()
            optimizer.step()
            if loss < minLoss:
                minLoss = loss
                minTheta = theta_init

        theta_init = minTheta
        self.set_sigma_net_params(theta_init.detach().cpu().numpy())
        self.posterior_mean = theta_init.detach().cpu().numpy()
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        hessian = self.hessian_approximator.compute(theta_init)
        reg_term = torch.eye(hessian.shape[0]).cuda() * 1e-2  # Tikhonov regularization
        hessian += reg_term
        self.posterior_cov = np.linalg.inv(hessian.detach().cpu().numpy())
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