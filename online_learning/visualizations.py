import matplotlib.pyplot as plt
import numpy as np


def show_atoms_img(atoms, figsize=(15, 8)):
    n_atoms = atoms.shape[0]
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


def show_dictionary_atoms_img(dict_learner, atom_h=28, atom_w=28, figsize=(15, 8)):
    atoms = np.reshape(dict_learner.components_,
                       (dict_learner.n_components, atom_h, atom_w))

    show_atoms_img(atoms, figsize=figsize)
