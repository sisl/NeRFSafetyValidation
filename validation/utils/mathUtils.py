import numpy as np

def is_positive_definite(matrix):
    matrix_copy = matrix.cpu().numpy()
    try:
        np.linalg.cholesky(matrix_copy)
        return True
    except np.linalg.LinAlgError:
        return False