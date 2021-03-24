import matplotlib.pyplot as plt
import numpy as np


def plot_losses_sparsities(xs, losses, sparsities, x_label):
    _, (ax0, ax1) = plt.subplots(ncols=2, figsize=(15, 5))
    ax0.loglog(xs, losses)
    ax0.set_xlabel(x_label)
    ax0.set_ylabel("L2 loss normalized")
    ax0.set_title(f"Impact of {x_label} on the L2 loss")

    ax1.loglog(xs, sparsities)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Sparsity ratio")
    ax1.set_title(f"Impact of {x_label} on the sparsity ratio")

    plt.show()


def plot_alphas_losses_sparsities(alphas, losses, sparsities):
    plot_losses_sparsities(alphas, losses, sparsities, "alpha")


def plot_n_components_losses_sparsities(n_components, losses, sparsities):
    plot_losses_sparsities(n_components, losses, sparsities, "n_components")


def show_dictionary_atoms_img(dict_learner, atom_h=28, atom_w=28, figsize=(15, 8)):
    atoms = np.reshape(dict_learner.components_,
                       (dict_learner.n_components, atom_h, atom_w))
    n_atoms = dict_learner.n_components
    n_imgs_per_row = 4
    nrows, ncols = int(np.ceil(n_atoms / n_imgs_per_row)
                       ), min(n_imgs_per_row, n_atoms)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx < n_atoms:
            ax.imshow(atoms[idx])
        else:
            ax.axis('off')
    fig.suptitle("Atoms")
    plt.show()
