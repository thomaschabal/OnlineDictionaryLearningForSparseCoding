import matplotlib.pyplot as plt
import numpy as np


# ===== Evaluate reconstruction error and dictionary stability =====


# Plot a line at the times corresponding to changes in the nature of the data
def plot_data_nature_changes(ax, data_nature_changes, ymax):
    for idx, change_x in enumerate(data_nature_changes):
        label = "Data nature change" if idx == 0 else None
        ax.vlines(change_x, 0, ymax, colors="red",
                  linestyles="dashed", label=label)


def plot_reconstruction_error(
        ax, times, reconstruction_errors, data_nature_changes_time=[], label_name="Test image", label_values=[]):

    # Plot the reconstruction error
    for idx, single_error_curve in enumerate(reconstruction_errors):
        label = f"{label_name} {label_values[idx]}" if len(
            label_values) > 0 else f"{label_name} {idx}"
        x_axis = times if type(times) == list else times[idx]
        ax.loglog(x_axis, single_error_curve, label=label)

    # If there are any changes in the signal, plot the times they appear
    if len(data_nature_changes_time) > 0:
        ymax = 1.015 * np.max(reconstruction_errors)
        plot_data_nature_changes(ax, data_nature_changes_time, ymax)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reconstruction error")
    ax.set_title("Reconstruction error through time")
    ax.legend()


def plot_dictionary_distances(
        ax, batches_seen, dictionary_atoms_distances, compute_atoms_distance_every, data_nature_changes_batches=[], label_name=None, label_values=[]):

    # Plot dictionary atoms distances
    for idx, dictionary_atoms_curve in enumerate(dictionary_atoms_distances):
        label = f"{label_name} {label_values[idx]}" if len(
            label_values) > 0 else label_name
        x_axis = batches_seen if type(
            batches_seen) == list else batches_seen[idx]
        ax.loglog(x_axis, dictionary_atoms_curve, label=label)

    # If there are any changes in the signal, plot the times they appear
    if len(data_nature_changes_batches) > 0:
        ymax = 1.015 * np.max(dictionary_atoms_distances)
        plot_data_nature_changes(ax, data_nature_changes_batches, ymax)

    ax.set_xlabel("Batches seen")
    ax.set_ylabel("Distance from the previous dictionary")
    ax.set_title(
        f"Evolution of the distance between dictionary atoms every {compute_atoms_distance_every} batches")
    ax.legend()


def plot_reconstruction_error_and_dictionary_distances(
        times, reconstruction_errors, batches_seen, dictionary_atoms_distances, compute_atoms_distance_every, data_nature_changes_time=[], data_nature_changes_batches=[], label_name="Test image", label_values=[]):
    _, (ax0, ax1) = plt.subplots(ncols=2, figsize=(15, 5))

    plot_reconstruction_error(
        ax0, times, reconstruction_errors, data_nature_changes_time, label_name, label_values)

    plot_dictionary_distances(
        ax1, batches_seen, dictionary_atoms_distances, compute_atoms_distance_every, data_nature_changes_batches, label_name, label_values)

    plt.show()
