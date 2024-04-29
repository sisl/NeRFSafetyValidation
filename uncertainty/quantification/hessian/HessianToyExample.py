import torch
from torch.autograd import functional as F

from uncertainty.quantification.hessian.HessianApproximator import HessianApproximator

# simple quadratic function
def func(input):
    return torch.sum(input**2)

# point to compute at
x = torch.randn(12, requires_grad=True)  # 12-dimensional vector

# actual hessian
actual_hessian = F.hessian(func, x)

for method in ['finite_difference', 'bfgs', 'regression_gradient']:
    approximator = HessianApproximator(func, method=method)
    approx_hessian = approximator.compute(x)
    diff = torch.norm(actual_hessian - approx_hessian)
    print("Actual Hessian: " + actual_hessian)
    print(f'Hessian approximated with {method}: ' + approx_hessian)
    print(f"Difference between actual Hessian and {method} approximation:", diff.item())