from sklearn.datasets import make_sparse_coded_signal
import arff
import numpy as np
import os

git_dir_name = 'OnlineDictionaryLearningForSparseCoding'
FACE_DATA_DIR = os.getcwd().split(git_dir_name)[
    0] + f'{git_dir_name}/datasets/FaceAll/'


def make_sparse_data(n_samples, n_features, n_components=15):
    X, _, _ = make_sparse_coded_signal(
        n_samples=n_samples, n_components=15, n_features=n_features,
        n_nonzero_coefs=10, random_state=42,
    )

    return X


def make_mnist():
    pass


def load_arff_data(file_path):
    dataset = arff.load(open(file_path, 'r'))
    np_array = np.array(dataset['data'])
    data = np_array[:, :-1]
    label = np_array[:, -1]
    return data.astype(float), label.astype(int)


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
