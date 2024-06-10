from uncertainty.quantification.hessian.methods import lbfgs, finite_difference, levenberg_marquardt, regression_gradient, regression_gradient_regularized


class HessianApproximator:
    def __init__(self, func, method='finite_difference', epsilon=1e-8, delta=1e-6, alpha=0.1, lmbda=0.01):
        """
        Initialize the Hessian approximator.

        Parameters:
        func: The function for which to compute the Hessian.
        method: The method to use for approximating the Hessian ('finite_difference' or 'lbfgs').
        epsilon: The small value used for finite differences (only used if method is 'finite_difference').
        """
        self.func = func
        self.method = method
        self.epsilon = epsilon
        self.alpha = alpha
        self.lmbda = lmbda
        self.delta = delta

    def compute(self, x):
        """
        Compute the Hessian of the function at a given point.

        Parameters:
        x: the point at which to compute the Hessian.

        Returns:
        the approximated Hessian matrix.
        """
        if self.method == 'finite_difference':
            return finite_difference(x, self.func, self.epsilon)
        elif self.method == 'bfgs':
            return lbfgs(x, self.func)
        elif self.method == 'regression_gradient':
            return regression_gradient(x, self.func, delta=self.delta)
        elif self.method == 'regression_gradient_regularized':
            return regression_gradient_regularized(x, self.func, delta=self.delta, alpha=self.alpha)
        elif self.method == 'levenberg_marquardt':
            return levenberg_marquardt(x, self.func, lmbda=self.lmbda)
        else:
            raise ValueError(f"Unknown method: {self.method}")
