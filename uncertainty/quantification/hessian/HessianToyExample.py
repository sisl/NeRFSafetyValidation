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

for method in ['finite_difference', 'bfgs', 'regression_gradient', 'regression_gradient_regularized', 'levenberg_marquardt']:
    approximator = HessianApproximator(func, method=method)
    approx_hessian = approximator.compute(x)
    diff_matrix = actual_hessian - approx_hessian
    diff = torch.norm(actual_hessian - approx_hessian)
    print("Actual Hessian: ")
    print(actual_hessian)
    print(f'Hessian approximated with {method}: ')
    print(approx_hessian)
    print(f"Difference between actual Hessian and {method} approximation:", diff.item())
    print(f"Element-wise difference between actual Hessian and {method} approximation:\n", diff_matrix)