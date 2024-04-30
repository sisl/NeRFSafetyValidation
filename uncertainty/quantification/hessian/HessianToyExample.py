import torch
from torch.autograd import functional as F
import numpy as np

from uncertainty.quantification.hessian.HessianApproximator import HessianApproximator

# simple quadratic function
def func(input):
    return torch.sum(input**2)

# point to compute at
x = torch.randn(12, requires_grad=True)  # 12-dimensional vector

# actual hessian
actual_hessian = F.hessian(func, x)

for method in ['finite_difference', 'bfgs']:
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

# regression gradient w/o regularization
deltas = np.logspace(-8, 1, num=1000)
best_delta = None
best_hessian = None
best_diff = float('inf')
for d in deltas:
    approximator = HessianApproximator(func, method='regression_gradient', delta=d)
    approx_hessian = approximator.compute(x)
    diff = torch.norm(actual_hessian - approx_hessian)

    if diff < best_diff:
        best_delta = d
        best_hessian = approx_hessian
        best_diff = diff

print("Actual Hessian: ")
print(actual_hessian)
print(f'Hessian approximated with regression_gradient: ')
print(best_hessian)
print(f"Difference between actual Hessian and regression_gradient approximation: ", best_diff.item())
print(f"Element-wise difference between actual Hessian and regression_gradient approximation:\n", best_hessian)
print("Best value for delta: ", best_delta)

# regression gradient w/ regularization
alphas = np.logspace(-8, 1, num=1000)
deltas = np.logspace(-8, 1, num=1000)
best_delta = None
best_alpha = None
best_hessian = None
best_diff = float('inf')
for a in alphas:
    for d in deltas:
        approximator = HessianApproximator(func, method='regression_gradient_regularized', delta=d, alpha=a)
        approx_hessian = approximator.compute(x)
        diff = torch.norm(actual_hessian - approx_hessian)

        if diff < best_diff:
            best_delta = d
            best_alpha = a
            best_hessian = approx_hessian
            best_diff = diff

print("Actual Hessian: ")
print(actual_hessian)
print(f'Hessian approximated with regression_gradient_regularized: ')
print(best_hessian)
print(f"Difference between actual Hessian and regression_gradient_regularized approximation: ", best_diff.item())
print(f"Element-wise difference between actual Hessian and regression_gradient_regularized approximation:\n", best_hessian)
print("Best value for delta: ", best_delta)
print("Best value for alpha: ", best_alpha)

# levenberg marquardt
lmbdas = np.logspace(-8, 1, num=1000)
best_lmbda = None
best_hessian = None
best_diff = float('inf')
for l in lmbdas:
    approximator = HessianApproximator(func, method='levenberg_marquardt', lmbda=l)
    approx_hessian = approximator.compute(x)
    diff = torch.norm(actual_hessian - approx_hessian)

    if diff < best_diff:
        best_lmbda = l
        best_hessian = approx_hessian
        best_diff = diff

print("Actual Hessian: ")
print(actual_hessian)
print(f'Hessian approximated with levenberg_marquardt: ')
print(best_hessian)
print(f"Difference between actual Hessian and levenberg_marquardt approximation: ", best_diff.item())
print(f"Element-wise difference between actual Hessian and levenberg_marquardt approximation:\n", best_hessian)
print("Best value for lmbda: ", best_lmbda)

