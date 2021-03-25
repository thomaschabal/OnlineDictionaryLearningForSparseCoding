import os
import numpy as np
from numpy.lib import recfunctions as rfn
from scipy.io.arff import loadarff
from sklearn.datasets import make_sparse_coded_signal
try:
    from torchvision.datasets import FashionMNIST
except ModuleNotFoundError:
    print("Cannot import FashionMNIST, install torchvision module.")


git_dir_name = 'OnlineDictionaryLearningForSparseCoding'
DATASETS_DIR = os.getcwd().split(git_dir_name)[0] + f'{git_dir_name}/datasets/'
FACE_DATA_DIR = DATASETS_DIR + 'FaceAll/'
FASHION_MNIST_DIR = DATASETS_DIR + 'FashionMNIST/'


#####################################################
#                                                   #
#   Available datasets:                             #
#       - Sparse data:      make_sparse_data        #
#       - Fashion MNIST:    make_fashion_mnist      #
#       - FaceAll:          make_faces              #
#                                                   #
#####################################################


def make_sparse_data(n_samples, n_features, n_components=15, random_state=None):
    X, _, _ = make_sparse_coded_signal(
        n_samples=n_samples, n_components=n_components, n_features=n_features,
        n_nonzero_coefs=10, random_state=random_state,
    )
    return X


def make_trendy_sparse(n_samples, n_features, a_coeff, n_components=15, random_state=None):
    X = make_sparse_data(n_samples, n_features, n_components, random_state).T
    X += np.linspace(0, a_coeff, n_features)[None, :]
    X *= a_coeff * np.arange(n_samples)[:, None]
    return X


# We return normalized images, which range in [0, 1]
def make_fashion_mnist(n_samples_train, n_samples_test=0):
    dataset = FashionMNIST(FASHION_MNIST_DIR, download=True)
    n_samples = min(len(dataset), n_samples_train + n_samples_test)
    X = np.array([np.array(dataset[idx][0]).flatten()
                  for idx in range(n_samples)])

    X_train = X[:n_samples_train] / 255
    X_test = X[n_samples_train:] / 255
    return X_train, X_test


def load_arff_data(file_path):
    data, __ = loadarff(file_path)
    data = rfn.structured_to_unstructured(data)
    return data[:, :-1].astype(float), data[:, -1].astype(int)


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

# Add audio dataset and other datasets
