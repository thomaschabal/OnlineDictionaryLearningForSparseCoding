import os
import numpy as np
import cv2
from numpy.lib import recfunctions as rfn
from scipy.io.arff import loadarff
from sklearn.datasets import make_sparse_coded_signal
try:
    from torchvision.datasets import FashionMNIST, CIFAR10
except ModuleNotFoundError:
    print("Cannot import FashionMNIST, install torchvision module.")
from .utils import grey_to_color_fashion_mnist_dataset, resize_fashion_mnist_dataset, load_arff_data


git_dir_name = 'OnlineDictionaryLearningForSparseCoding'
DATASETS_DIR = os.getcwd().split(git_dir_name)[0] + f'{git_dir_name}/datasets/'
FACE_DATA_DIR = DATASETS_DIR + 'FaceAll/'
FASHION_MNIST_DIR = DATASETS_DIR + 'FashionMNIST/'
CIFAR10_DIR = DATASETS_DIR + 'CIFAR10/'

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat",
                 "deer", "dog", "frog", "horse", "ship", "truck"]

#########################################################
#                                                       #
#   Available datasets:                                 #
#       - Sparse data:          make_sparse_data        #
#       - Sparse trendy data:   make_trendy_sparse      #
#       - Fashion MNIST:        make_fashion_mnist      #
#       - CIFAR10:              make_cifar10            #
#       - FaceAll:              make_faces              #
#                                                       #
#########################################################


# ===== Sparse datasets =====

def make_sparse_data(n_samples: int, n_features: int, n_components=15, random_state=None):
    X, _, _ = make_sparse_coded_signal(
        n_samples=n_samples, n_components=n_components, n_features=n_features,
        n_nonzero_coefs=10, random_state=random_state,
    )
    return X


def make_trendy_sparse(n_samples: int, n_features: int, a_coeff: float, n_components=15, random_state=None):
    X = make_sparse_data(n_samples, n_features, n_components, random_state).T
    X += np.linspace(0, a_coeff, n_features)[None, :]
    X *= a_coeff * np.arange(n_samples)[:, None]
    return X


# ===== FashionMNIST dataset =====

# We return normalized images, which range in [0, 1]
def make_fashion_mnist(n_samples_train: int, n_samples_test=0):
    dataset = FashionMNIST(FASHION_MNIST_DIR, download=True)
    n_samples = min(len(dataset), n_samples_train + n_samples_test)
    X = np.array([np.array(dataset[idx][0]).flatten()
                  for idx in range(n_samples)])

    X_train = X[:n_samples_train] / 255
    X_test = X[n_samples_train:] / 255
    return X_train, X_test


def make_color_fashion_mnist(n_samples_train: int, n_samples_test=0):
    X_train, X_test = make_fashion_mnist(n_samples_train, n_samples_test)

    X_train_color = grey_to_color_fashion_mnist_dataset(X_train)
    X_test_color = grey_to_color_fashion_mnist_dataset(X_test)

    return X_train_color, X_test_color


def make_fashion_mnist_matching_cifar10(n_samples_train: int, n_samples_test=0):
    X_train, X_test = make_color_fashion_mnist(n_samples_train, n_samples_test)

    X_train = resize_fashion_mnist_dataset(X_train)
    X_test = resize_fashion_mnist_dataset(X_test)

    return X_train, X_test


# ===== FaceAll dataset =====

def make_faces():
    if not os.path.exists(FACE_DATA_DIR):
        download_path = "https://timeseriesclassification.com/description.php?Dataset=FaceAll"
        raise Exception(
            f"Please download the dataset from {download_path} and put it" +
            " in a 'datasets' folder at the root of this repository.")

    data_file_train = FACE_DATA_DIR + "FaceAll_TRAIN.arff"
    train_data, train_label = load_arff_data(data_file_train)

    data_file_test = FACE_DATA_DIR + "FaceAll_TEST.arff"
    test_data, test_label = load_arff_data(data_file_test)

    data = np.concatenate([train_data, test_data])
    label = np.concatenate([train_label, test_label])

    return data, label


# ===== CIFAR10 dataset =====

def make_cifar10(n_samples_train: int, n_samples_test=0, specify_class=None):
    dataset = CIFAR10(CIFAR10_DIR, download=True)
    n_samples = min(len(dataset), n_samples_train + n_samples_test)

    if specify_class is not None:
        try:
            class_idx = CIFAR_CLASSES.index(specify_class)
        except Exception as e:
            raise Exception(e, "Available classes: ", CIFAR_CLASSES)

        X = np.array([np.array(val).flatten()
                     for val, lbl in dataset if lbl == class_idx])
        X = X[:n_samples]
    else:
        X = np.array([np.array(dataset[idx][0]).flatten()
                      for idx in range(n_samples)])

    X_train = X[:n_samples_train] / 255
    X_test = X[n_samples_train:] / 255
    return X_train, X_test


# ===== FashionMNIST + CIFAR10 dataset =====

def make_mix_fashion_mnist_cifar10(n_samples_mnist: tuple, n_samples_cifar: tuple, specify_class_cifar=None):
    assert type(n_samples_mnist) == tuple, \
        "n_samples_mnist should be a tuple of size 2: (size_train, size_test)"
    assert type(n_samples_cifar) == tuple, \
        "n_samples_cifar should be a tuple of size 2: (size_train, size_test)"
    assert len(n_samples_mnist) == 2, \
        "The number of samples for FashionMNIST should be a tuple of length 2 :(size_train, size_test)"
    assert len(n_samples_cifar) == 2, \
        "The number of samples for CIFAR10 should be a tuple of length 2 :(size_train, size_test)"

    n_samples_mnist_train, n_samples_mnist_test = n_samples_mnist
    n_samples_cifar_train, n_samples_cifar_test = n_samples_cifar

    X_train_mnist, X_test_mnist = make_fashion_mnist_matching_cifar10(
        n_samples_mnist_train, n_samples_mnist_test)
    X_train_cifar, X_test_cifar = make_cifar10(
        n_samples_cifar_train, n_samples_cifar_test, specify_class_cifar)

    X_train = np.concatenate((X_train_mnist, X_train_cifar), axis=0)
    X_test = np.concatenate((X_test_mnist, X_test_cifar), axis=0)

    return X_train, X_test


# Add audio dataset and other datasets
