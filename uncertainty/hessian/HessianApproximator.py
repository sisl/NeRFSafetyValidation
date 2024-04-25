from uncertainty.hessian.methods import bfgs, finite_difference


class HessianApproximator:
    def __init__(self, func, method='finite_difference', epsilon=1e-4):
        """
        Initialize the Hessian approximator.

        Parameters:
        func: The function for which to compute the Hessian.
        method: The method to use for approximating the Hessian ('finite_difference' or 'bfgs').
        epsilon: The small value used for finite differences (only used if method is 'finite_difference').
        """
        self.func = func
        self.method = method
        self.epsilon = epsilon

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
            return bfgs(x)
        else:
            raise ValueError(f"Unknown method: {self.method}")
