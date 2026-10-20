import numpy as np


def pairwise_sq_dists(X, C):
    # TODO: ||x||^2 + ||c||^2 - 2 x.c via one matrix product, clipped at 0

    return ((X[:, None, :] - C[None, :, :])**2).sum(axis=-1)


def assign_clusters(X, C):
    # TODO: argmin over centroids
    return np.argmin(pairwise_sq_dists(X, C), axis=-1)
    

def update_centroids(X, labels, C):
    # TODO: mean of each cluster's points; empty clusters keep their centroid
    for k in range(len(C)):
        pts = X[labels == k]

        if pts.size > 0:
            C[k] = pts.mean(axis=0)
    
    return C


def kmeans(X, C0, max_iter=100, tol=1e-6):
    # TODO: iterate until the largest centroid movement < tol; return (C, labels, n_iter)
    C = C0.copy()

    n_iter = 0
    labels = None
    while n_iter < max_iter:
        prev_C = C.copy()
        n_iter += 1

        labels = assign_clusters(X, C)
        C = update_centroids(X, labels, C)

        if np.allclose(C, prev_C, atol=tol):
            return C, labels, n_iter
    
    return C, labels, n_iter


def inertia(X, C, labels):
    dists = pairwise_sq_dists(X,C)

    return (dists[np.arange(len(labels)), labels]).sum()
