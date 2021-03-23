from tqdm import tqdm
from sklearn.decomposition import DictionaryLearning

from .metrics import l2_normalized_error, sparsity_ratio


def get_loss_sparse_ratio_from_dict_learner(dict_learner, X):
    X_transformed = dict_learner.fit_transform(X)
    X_hat = X_transformed @ dict_learner.components_

    loss = l2_normalized_error(X, X_hat)
    sparse_ratio = sparsity_ratio(X_transformed)
    return loss, sparse_ratio


def eval_parameter(parameters, X, dict_learner_fct):
    losses = []
    sparse_ratios = []
    for param in tqdm(parameters):
        dict_learner = dict_learner_fct(param)

        loss, sparse_ratio = get_loss_sparse_ratio_from_dict_learner(
            dict_learner, X)
        losses.append(loss)
        sparse_ratios.append(sparse_ratio)

    return losses, sparse_ratios


def eval_n_components(numbers_components, X):
    def dict_learner_fct(n_components): return DictionaryLearning(
        n_components=n_components, transform_algorithm='lasso_lars',
        random_state=42,
    )

    return eval_parameter(numbers_components, X, dict_learner_fct)


def eval_alpha(alphas, X):
    def dict_learner_fct(alpha): return DictionaryLearning(
        n_components=15, transform_algorithm='lasso_lars', random_state=42,
        alpha=alpha,
    )

    return eval_parameter(alphas, X, dict_learner_fct)


def eval_tol(tols, X):
    def dict_learner_fct(tol): return DictionaryLearning(
        n_components=15, transform_algorithm='lasso_lars', random_state=42,
        tol=tol,
    )

    return eval_parameter(tols, X, dict_learner_fct)


def eval_fit_algos(fit_algos, X):
    pass


def eval_transform_algos(transform_algos, X):
    pass


def eval_dict_init(dict_inits, X):
    # Eval online performances
    pass
