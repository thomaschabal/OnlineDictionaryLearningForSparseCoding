import numpy as np


def l2_images_error(img, img_hat):
    return np.linalg.norm(img - img_hat)


def distance_between_atoms(atoms1, atoms2):
    distance = np.mean(np.linalg.norm(atoms1 - atoms2, axis=1))
    return distance
