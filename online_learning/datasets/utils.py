import numpy as np
import cv2
from numpy.lib import recfunctions as rfn
from scipy.io.arff import loadarff


# ==== Colorize functions for FashionMNIST

def grey_to_color_fashion_mnist_feature(feat):
    feat = np.reshape(feat, (28, 28))[:, :, np.newaxis]
    feat = np.concatenate((feat, feat, feat), axis=2).flatten()
    return feat


def grey_to_color_fashion_mnist_dataset(X):
    return np.array([grey_to_color_fashion_mnist_feature(feat) for feat in X])


# ==== Resize functions for FashionMNIST to match with CIFAR10

def resize_fashion_mnist_feat(feat, dest_size=(32, 32)):
    img = np.reshape(feat, (28, 28, 3))
    img = cv2.resize(img, dest_size, cv2.INTER_CUBIC)
    feat = img.flatten()
    return feat


def resize_fashion_mnist_dataset(X, dest_size=(32, 32)):
    return np.array([resize_fashion_mnist_feat(feat, dest_size) for feat in X])
