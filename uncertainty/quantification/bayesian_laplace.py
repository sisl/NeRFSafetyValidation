import numpy as np
from scipy.optimize import minimize

class BayesianLaplace:
    def __init__(self, model, prior_mean, prior_std):
        """
        Initialize the BayesianLaplace class.

        Parameters:
        model (object): The model to be used.
        prior_mean (float): The mean of the prior distribution.
        prior_std (float): The standard deviation of the prior distribution.
        """
        
        self.model = model
        self.prior_mean = prior_mean
        self.prior_std = prior_std

    def log_prior(self, theta):
        return -0.5 * np.sum((theta - self.prior_mean)**2 / self.prior_std**2)

    def log_likelihood(self, theta, X, y):
        self.model.set_params(theta)
        y_pred = self.model.predict(X)
        #print(y_pred)
        return -0.5 * np.sum((y - y_pred)**2)

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
        theta_init = self.model.get_params()
        res = minimize(self.negative_log_posterior, theta_init, args=(X, y), jac=self.grad_negative_log_posterior, method='BFGS')
        self.model.set_params(res.x)
        self.posterior_mean = res.x
        self.posterior_cov = np.linalg.inv(self.hessian_negative_log_posterior(res.x, X, y))
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_posterior_mean(self):
        return self.posterior_mean

    def get_posterior_cov(self):
        return self.posterior_cov