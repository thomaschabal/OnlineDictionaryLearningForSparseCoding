from tqdm import tqdm
from sklearn.decomposition import MiniBatchDictionaryLearning
import numpy as np
import time

from .metrics import distance_between_atoms
from .robustness import loader
from .visualizations import show_dictionary_atoms_img
from .plots import plot_reconstruction_error_and_dictionary_distances


def eval_impact_parameters_on_reconstruction(
        X: np.ndarray, X_test: np.ndarray, list_n_atoms: list,
        list_batch_sizes: list, list_alphas: list,
        display_atoms_every=10, color=True, atom_h=32, atom_w=32,
        display_intermediate=True, label_name=None, label_values=[]):
    """
        X: array of shape (num_samples, feature_size)
        X_test: array of shape (1, feature_size)
        list_n_atoms: list of int, list of number of components to evaluate reconstruction on
        list_batch_sizes: list of int, list of batch sizes to use for on-the-fly partial fit
        list_alphas: list of float, list of alpha values (sparsity parameter) to use for on-the-fly partial fit
    """
    times, reconstruction_errors = [], []
    batches_seen, dictionary_atoms_distances = [], []

    # For the given parameters, we initialize a dictionary learning
    for n_atoms in list_n_atoms:
        for batch_size in list_batch_sizes:
            for alpha in list_alphas:
                times_for_params, reconstruction_errors_for_params = [], []
                batches_seen_for_params, dictionary_atoms_distances_for_params = [], []

                clf = MiniBatchDictionaryLearning(n_components=n_atoms,
                                                  batch_size=batch_size,
                                                  alpha=alpha,
                                                  transform_algorithm='lasso_lars',
                                                  verbose=False)

                former_atoms = np.zeros((n_atoms, X_test.shape[1]))

                start = time.time()
                # For every batch of image, we compute a partial fit of the dictionary
                for i, sample in tqdm(loader(X, batch_size), total=X.shape[0] // batch_size):
                    clf.partial_fit(sample)

                    # We then measure reconstruction error and the atoms distances between each iteration
                    reconstruction_error = np.linalg.norm(
                        X_test - clf.transform(X_test).dot(clf.components_))
                    reconstruction_errors_for_params.append(
                        reconstruction_error)
                    times_for_params.append(time.time() - start)

                    # We also compute the distance variation of atoms between two batches
                    distance_from_prev_dict = distance_between_atoms(
                        former_atoms, clf.components_)
                    former_atoms = np.copy(clf.components_)
                    dictionary_atoms_distances_for_params.append(
                        distance_from_prev_dict)
                    batches_seen_for_params.append(i)

                    # Occasionally, we display the computed atoms of the dictionary
                    display_atoms_cond = i % display_atoms_every == display_atoms_every - 1
                    if display_atoms_cond and display_intermediate:
                        print("=" * 20, "\n", "Batch", i)
                        print("Distance between current and previous atoms:",
                              distance_from_prev_dict)
                        show_dictionary_atoms_img(
                            clf, color=color, atom_h=atom_h, atom_w=atom_w)

                reconstruction_errors.append(reconstruction_errors_for_params)
                dictionary_atoms_distances.append(
                    dictionary_atoms_distances_for_params)
                times.append(times_for_params)
                batches_seen.append(batches_seen_for_params)

    reconstruction_errors = np.array(reconstruction_errors)
    dictionary_atoms_distances = np.array(dictionary_atoms_distances)
    times = np.array(times)
    batches_seen = np.array(batches_seen)

    # We eventually plot the reconstruction error and the evolution of atoms distances
    plot_reconstruction_error_and_dictionary_distances(
        times, reconstruction_errors, batches_seen, dictionary_atoms_distances, 1, label_name=label_name, label_values=label_values)
