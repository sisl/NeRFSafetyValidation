import torch
from torch.autograd import functional as F

from uncertainty.quantification.hessian.HessianApproximator import HessianApproximator

# simple quadratic function
def func(input):
    x, y = input
    return x**2 + y**2

# point to compute at
x = torch.tensor([1.0, 2.0], requires_grad=True)

# actual hessian
actual_hessian = F.hessian(func, x)

for method in ['finite_difference', 'bfgs', 'regression_gradient']:
    approximator = HessianApproximator(func, method=method)
    approx_hessian = approximator.compute(x)
    diff = torch.norm(actual_hessian - approx_hessian)
    print(f"Difference between actual Hessian and {method} approximation:", diff.item())