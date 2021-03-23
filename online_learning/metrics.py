import numpy as np


def l2_normalized_error(X, X_hat):
    return np.mean(np.sum((X_hat - X) ** 2, axis=1) / np.sum(X ** 2, axis=1))


def sparsity_ratio(X):
    return np.mean(X == 0)
