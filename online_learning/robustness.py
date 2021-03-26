import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import MiniBatchDictionaryLearning
from .metrics import distance_between_atoms
from .visualizations import show_dictionary_atoms_img
from .plots import plot_reconstruction_error_and_dictionary_distances


def loader(X, batch_size):
    for j, i in enumerate(range(0, len(X), batch_size)):
        try:
            yield j, X[i: i + batch_size]
        except IndexError:
            yield j, X[i:]


def study_dictionary_convergence_and_reconstruction_for_images(
        X: np.ndarray, X_test: np.ndarray, n_atoms=10, batch_size=30, data_nature_changes=[], compute_atoms_distance_every=10, color=True, atom_h=32, atom_w=32, display_intermediate=True):
    """
        X: array of shape (num_samples, feature_size)
        X_test: array of shape (num_samples, feature_size)
    """
    times, reconstruction_errors = [], []
    dictionary_atoms_distances, batches_seen = [], []
    data_nature_changes_batches = [
        size // batch_size for size in data_nature_changes]
    data_nature_changes_time = []

    clf = MiniBatchDictionaryLearning(n_components=n_atoms,
                                      batch_size=batch_size,
                                      transform_algorithm='lasso_lars',
                                      verbose=False)

    former_atoms = np.zeros((n_atoms, X_test.shape[1]))

    start = time.time()
    # For every batch of image, compute a partial fit of the dictionary
    for i, sample in tqdm(loader(X, batch_size), total=X.shape[0] // batch_size):
        clf.partial_fit(sample)

        # We then measure reconstruction error and the atoms distances between each iteration
        reconstruction_error = np.array([np.linalg.norm(
            test_x - clf.transform(test_x).dot(clf.components_)) for test_x in X_test])
        reconstruction_errors.append(reconstruction_error)
        times.append(time.time() - start)

        # We compute the data nature change time if there is any
        nb_of_current_changes = len(data_nature_changes_time)
        if nb_of_current_changes < len(data_nature_changes):
            # Data nature change at current batch
            if data_nature_changes[nb_of_current_changes] <= i * batch_size:
                data_nature_changes_time.append(time.time() - start)

        atoms_distance_computation_cond = i % compute_atoms_distance_every == compute_atoms_distance_every - 1
        if atoms_distance_computation_cond:
            distance_from_prev_dict = distance_between_atoms(
                former_atoms, clf.components_)
            former_atoms = np.copy(clf.components_)
            dictionary_atoms_distances.append(distance_from_prev_dict)

            batches_seen.append(i)

            if display_intermediate:
                print("=" * 20, "\n", "Batch", i)
                print("Distance between current and previous atoms:",
                      distance_from_prev_dict)
                show_dictionary_atoms_img(
                    clf, color=color, atom_h=atom_h, atom_w=atom_w)

    # We eventually plot the reconstruction error and the evolution of atoms distances
    reconstruction_errors = np.array(reconstruction_errors).T
    dictionary_atoms_distances = np.array(dictionary_atoms_distances)
    plot_reconstruction_error_and_dictionary_distances(
        times, reconstruction_errors, batches_seen, dictionary_atoms_distances, compute_atoms_distance_every, data_nature_changes_time, data_nature_changes_batches)
