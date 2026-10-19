import numpy as np


def pairwise_sq_dists(X, C):
    # TODO: ||x||^2 + ||c||^2 - 2 x.c via one matrix product, clipped at 0
    return ((X[:, None, :] - C[None, :, :])**2).sum(axis=-1)


def assign_clusters(X, C):
    dists = pairwise_sq_dists(X, C)
    return np.argmin(dists, axis=-1) #[number of points, 1]


def update_centroids(X, labels, C):
    # TODO: mean of each cluster's points; empty clusters keep their centroid
    new_c = C.copy()

    for k in range(len(C)):
        pts = X[labels == k]
    
        if pts.size > 0:
            new_c[k] = pts.mean(axis=0)
    
    return new_c


def kmeans(X, C0, max_iter=100, tol=1e-6):
    C = C0.copy()

    i = 0
    labels = None
    while i < max_iter:
        i += 1
        prev_C = C.copy()

        labels = assign_clusters(X, C)
        C = update_centroids(X, labels, C)

        if np.allclose(prev_C, C, atol=tol):
            return C, labels, i
    
    return C, labels, i


def inertia(X, C, labels):
    dists = pairwise_sq_dists(X, C)
    return np.sum(dists[np.arange(len(labels)), labels])

