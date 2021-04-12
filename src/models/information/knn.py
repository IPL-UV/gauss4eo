import numpy as np
from scipy.special import gamma, psi
from sklearn.neighbors import NearestNeighbors
from typing import Optional
from sklearn.utils import check_random_state, check_array
from typing import Optional, Union, Iterable
import time


def knn_entropy(X: np.ndarray, n_neighbors: int = 5, **kwargs) -> float:
    n_samples, d_dimensions = X.shape

    # volume of unit ball in d^n
    vol = (np.pi ** (0.5 * d_dimensions)) / gamma(0.5 * d_dimensions + 1)

    # 1. Calculate the K-nearest neighbors
    distances = knn_distance(X, n_neighbors=n_neighbors + 1, **kwargs)

    # return distance to kth nearest neighbor
    distances = distances[:, -1]

    # add error margin to avoid zeros
    distances += np.finfo(X.dtype).eps

    p_k_hat = (
        (n_neighbors / (n_samples - 1.0))
        * (1.0 / vol)
        * (1.0 / distances ** d_dimensions)
    )

    log_p_k_hat = np.log(p_k_hat)

    h_k_hat = log_p_k_hat.sum() / (-1.0 * n_samples)

    return h_k_hat

    # # estimation
    # return float(
    #     d_dimensions * np.mean(np.log(distances) )
    #     + np.log(vol)
    #     + psi(n_samples)
    #     - psi(n_neighbors)
    # )


def knn_total_corr(variables: Iterable, n_neighbours: int = 1, **kwargs) -> float:

    marginal_h = 0.0
    for ivar in variables:
        marginal_h += knn_entropy(ivar, n_neighbors=n_neighbours, **kwargs)

    # H = sum h_i - H(X, ...)
    h = marginal_h - knn_entropy(
        np.concatenate(variables, axis=1), n_neighbors=n_neighbours, **kwargs
    )

    return float(h)


def knn_mutual_info(
    X: np.ndarray, Y: np.ndarray, n_neighbours: int = 1, **kwargs
) -> float:
    t0 = time.time()

    # calculate the marginal entropy
    H_x = knn_entropy(X, n_neighbors=n_neighbours, **kwargs)
    H_y = knn_entropy(Y, n_neighbors=n_neighbours, **kwargs)
    H_marg = H_x + H_y

    # H = sum h_i - H(X, ...)
    H_xy = knn_entropy(np.hstack([X, Y]), n_neighbors=n_neighbours, **kwargs)
    knn_mi = H_marg - H_xy

    return {
        "knn_mi": knn_mi,
        "knn_H_joint": H_xy,
        "knn_H_marg": H_marg,
        "knn_H_x": H_x,
        "knn_H_y": H_y,
        "knn_time": time.time() - t0,
    }


# volume of unit ball
def volume_unit_ball(d_dimensions: int, radii: int, norm=2) -> float:
    """Volume of the d-dimensional unit ball

    Parameters
    ----------
    d_dimensions : int
        Number of dimensions to estimate the volume

    radii : int,

    norm : int, default=2
        The type of ball to get the volume.
        * 2 : euclidean distance
        * 1 : manhattan distance
        * 0 : chebyshev distance

    Returns
    -------
    vol : float
        The volume of the d-dimensional unit ball
    """

    # get ball
    if norm == 0:
        b = float("inf")
    elif norm == 1:
        b = 1.0
    elif norm == 2:
        b = 2.0
    else:
        raise ValueError(f"Unrecognized norm: {norm}")

    return (np.pi ** (0.5 * d_dimensions)) ** d_dimensions / gamma(b / d_dimensions + 1)


# KNN Distances
def knn_distance(
    X: np.ndarray,
    n_neighbors: int = 20,
    **kwargs,
) -> np.ndarray:
    """Light wrapper around sklearn library.

    Parameters
    ----------
    X : np.ndarray, (n_samples x d_dimensions)
        The data to find the nearest neighbors for.

    n_neighbors : int, default=20
        The number of nearest neighbors to find.

    algorithm : str, default='brute',
        The knn algorithm to use.
        ('brute', 'ball_tree', 'kd_tree', 'auto')

    n_jobs : int, default=-1
        The number of cores to use to find the nearest neighbors

    kwargs : dict, Optional
        Any extra keyword arguments.

    Returns
    -------
    distances : np.ndarray, (n_samples x d_dimensions)
    """
    clf_knn = NearestNeighbors(n_neighbors=n_neighbors, **kwargs)

    clf_knn.fit(X)

    dists, _ = clf_knn.kneighbors(X)

    return dists
