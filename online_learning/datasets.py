from sklearn.datasets import make_sparse_coded_signal


def make_sparse_data(n_samples, n_features, n_components=15):
    X, _, _ = make_sparse_coded_signal(
        n_samples=n_samples, n_components=15, n_features=n_features,
        n_nonzero_coefs=10, random_state=42,
    )

    return X


def make_mnist():
    pass

# Add audio dataset and other datasets
