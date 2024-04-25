import torch
from scipy.optimize import fmin_bfgs


def finite_difference(x, func, epsilon):
        """
        Approximate the Hessian using the finite difference method.

        Parameters:
        x: the point at which to compute the Hessian.

        Returns:
        hessian: the approximated Hessian matrix.
        """
        n = x.numel()
        size = x.size()
        hessian = torch.zeros(n, n)
        f_x = func(x)
        grad_x = torch.autograd.grad(f_x, x, create_graph=True)[0]

        for i in range(n):
            x_i = x.clone()
            x_i[i] += epsilon
            f_x_i = func(x_i)
            grad_x_i = torch.autograd.grad(f_x_i, x_i, create_graph=True)[0]
            hessian[i] = (grad_x_i - grad_x) / epsilon

        return hessian.view(*size, *size)

def bfgs(x):
    """
    Approximate the Hessian using the BFGS method (scipy optimizer).

    Parameters:
    x: the point at which to compute the Hessian.

    Returns:
    hessian: the approximated Hessian matrix.
    """
    x = x.detach().numpy()
    func = lambda x: func(torch.tensor(x)).item()
    grad = lambda x: torch.autograd.grad(func(torch.tensor(x)), torch.tensor(x))[0].numpy()
    hessian_inv = fmin_bfgs(func, x, fprime=grad, disp=False, full_output=True)[3]
    hessian = torch.inverse(torch.tensor(hessian_inv, dtype=torch.float))
    return hessian