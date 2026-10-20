import numpy as np


def pairwise_sq_dists(X, C):
    # TODO: ||x||^2 + ||c||^2 - 2 x.c via one matrix product, clipped at 0
    return np.sum((X[:, None, :] - C[None, :, :])**2, axis=-1)


def assign_clusters(X, C):
    # TODO: argmin over centroids
    dists = pairwise_sq_dists(X, C)

    return np.argmin(dists, axis=-1)


def update_centroids(X, labels, C):
    # TODO: mean of each cluster's points; empty clusters keep their centroid
    new_C = C.copy()
    for k in range(len(C)):
        chosen = X[labels == k]

        if chosen.size > 0:
            new_C[k] = np.mean(chosen, axis=0)
    
    return new_C


def kmeans(X, C0, max_iter=100, tol=1e-6):
    # TODO: iterate until the largest centroid movement < tol; return (C, labels, n_iter)
    n_iter = 0
    C = C0.copy()
    labels = None
    while n_iter < max_iter:
        prev_C = C.copy()
        n_iter += 1

        labels = assign_clusters(X, C)
        C = update_centroids(X, labels, C)

        if np.linalg.norm(C - prev_C) < tol:
            return C, labels, n_iter
    
    return C, labels, n_iter


def inertia(X, C, labels):
    dists = pairwise_sq_dists(X, C)

    return np.sum(dists[np.arange(len(labels)), labels])
