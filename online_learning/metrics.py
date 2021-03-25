import numpy as np


def l2_normalized_error(X, X_hat):
    return np.mean(np.sum((X_hat - X) ** 2, axis=1) / np.sum(X ** 2, axis=1))


def l2_images_error(img, img_hat):
    return np.linalg.norm(img - img_hat)


def sparsity_ratio(X):
    return np.mean(X == 0)


def distance_between_atoms(atoms1, atoms2):
    distance = np.mean(np.linalg.norm(atoms1 - atoms2, axis=1))
    return distance


def distance_between_dictionaries(dict1, dict2):
    return distance_between_atoms(dict1.components_, dict2.components_)
