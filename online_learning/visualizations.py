import matplotlib.pyplot as plt


def plot_losses_sparsities(xs, losses, sparsities, x_label):
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(15, 5))
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
