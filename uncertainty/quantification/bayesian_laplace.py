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

    def fit(self, X, y):
        theta_init = np.concatenate([param.detach().cpu().numpy().ravel() for param in self.model.sigma_net.parameters()])
        theta_init = torch.tensor(theta_init, requires_grad=True).cuda()
        X = torch.tensor(X).cuda()
        y = torch.tensor(y).cuda()

        # spatial perturbations in input 
        num_perturbations = 3
        perturbation_scale = 0.3
        perturbations = torch.randn((num_perturbations,) + X.shape, device='cuda') * perturbation_scale
        X_perturbed = X.unsqueeze(0) + perturbations

        minLoss, minTheta = float('inf'), theta_init
        for X_p in X_perturbed:
            theta = theta_init.clone().detach().requires_grad_(True)  # Create a copy of theta_init for each X_p
            optimizer = torch.optim.SGD([theta], lr=self.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
            for _ in range(1000):
                optimizer.zero_grad()
                loss = self.negative_log_posterior(theta, X_p, y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                if loss < minLoss:
                    minLoss = loss
                    minTheta = theta
                    
            # delete tensors and free up GPU memory
            del X_p, theta, optimizer, scheduler, loss
            torch.cuda.empty_cache()

        print("CHECK LOSS & THETA:")
        print(minLoss, minTheta)
        self.set_sigma_net_params(minTheta.detach().cpu().numpy())
        self.posterior_mean = minTheta.detach().cpu().numpy()
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        hessian = self.hessian_approximator.compute(minTheta)
        reg_term = torch.eye(hessian.shape[0]).cuda() * 1e-2  # Tikhonov regularization
        hessian += reg_term
        self.posterior_cov = np.linalg.inv(hessian.detach().cpu().numpy())
                    
        # delete tensors and free up GPU memory
        del theta_init, X, y, hessian, reg_term
        torch.cuda.empty_cache()
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