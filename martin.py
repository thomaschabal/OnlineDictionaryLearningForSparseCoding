import numpy as np
from online_learning.datasets import make_faces, make_sparse_data, make_trendy_sparse
from sklearn.decomposition import MiniBatchDictionaryLearning, DictionaryLearning
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def loader(X, batch_size):
    for j, i in enumerate(range(0, len(X), batch_size)):
        try:
            yield j, X[i:i+batch_size]
        except IndexError:
            yield j, X[i:]


# X = make_trendy_sparse(5000, 100, 2.0, 10, None)
X = make_sparse_data(n_samples=5000,
                     n_features=100,  # length of the signal
                     n_components=10,
                     random_state=None).T
# X, __ = make_faces()


## dictionary stabilisation

out = []

for j in range(3,8):
    n_components = j
    batch_size = 5
    verbose = 0
    d = np.random.randn(n_components, X.shape[1])
    clf = MiniBatchDictionaryLearning(n_components=n_components,
                                      batch_size=batch_size,
                                      # dict_init=d,
                                      verbose=verbose)

    for i, sample in loader(X, batch_size):
        clf.partial_fit(sample)
        if verbose:
            print()
        if np.allclose(d, clf.components_, atol=1e-4):  # dict == previous dict
            # pass
            print(i)
            out.append([j, i*batch_size])
            break
        d = clf.components_.copy()

plt.scatter(*np.array(out).T)
plt.xlabel("Parameter")
plt.ylabel("Nb of samples until stable")
plt.show()


## atoms stabilisation

n_components = 10
batch_size = 10
verbose = 0
d = np.random.randn(n_components, X.shape[1])
clf = MiniBatchDictionaryLearning(n_components=n_components,
                                  batch_size=batch_size,
                                  # dict_init=d,
                                  verbose=verbose)

out = []
tolerance = 1e-2

for i, sample in loader(X, batch_size):
    clf.partial_fit(sample)
    if verbose:
        print()
    out.append(np.sum([np.allclose(d[j], clf.components_[j], atol=tolerance) for j in range(n_components)]))
    d = clf.components_.copy()

plt.plot(out, 'o')
plt.xlabel("Iteration")
plt.ylabel("Number of stable atoms")
plt.title(f"Tolerance {tolerance}")
plt.show()


## convergence of reconstruction

out = []
test_x = X[0].reshape(1, -1)
X = X[1:]

n_components = 15
batch_size = 1
verbose = 0
clf = MiniBatchDictionaryLearning(n_components=n_components,
                                  batch_size=batch_size,
                                  transform_algorithm='lasso_lars',
                                  # dict_init=d,
                                  verbose=verbose)

for i, sample in loader(X, batch_size):
    clf.partial_fit(sample)
    if verbose:
        print()
    s = np.linalg.norm(test_x - clf.transform(test_x).dot(clf.components_))
    out.append(s)
    if s < 1e-6:
        break

plt.plot(out)
plt.xlabel("Iteration")
plt.ylabel("Reconstruction error")
plt.show()


## display dictionary

n_components = 10
batch_size = 10
verbose = 0
clf = MiniBatchDictionaryLearning(n_components=n_components,
                                  batch_size=batch_size,
                                  transform_algorithm='lasso_lars',
                                  # dict_init=d,
                                  verbose=verbose)

out = []
sample_every = 3
if n_components%2 == 0:
    fig, axs = plt.subplots(n_components//2, 2)
else:
    fig, axs = plt.subplots(n_components//2+1, 2)
lines = [ax.plot(X[1])[0] for ax in axs.flatten()]

for i, sample in loader(X, batch_size):
    clf.partial_fit(sample)
    if verbose:
        print()
    if i%sample_every == 0:
        out.append(clf.components_.copy())


def animate(i):
    fig.suptitle(f"Iteration {i*sample_every}")
    for line, component in zip(lines, out[i]):
        line.set_ydata(component)
    return lines,


ani = animation.FuncAnimation(fig, animate, frames=len(out), repeat=True)
plt.show()
