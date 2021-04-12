from typing import Callable

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.preprocessing import KernelCenterer
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from typing import Dict, Optional
import time
from scipy.spatial.distance import squareform, pdist, cdist


def estimate_sigma(
    X: np.ndarray,
    percent: int = 50,
    heuristic: bool = False,
) -> float:

    # get the squared euclidean distances

    kth_sample = int((percent / 100) * X.shape[0])
    dists = np.sort(squareform(pdist(X, "sqeuclidean")))[:, kth_sample]

    sigma = np.median(dists)

    if heuristic:
        sigma = np.sqrt(sigma / 2)
    return sigma


def init_rbf_kernel(n_sub_samples: int = 1_000, seed: int = 123):
    def k(X, Y):

        # subsample data

        sigma = sigma_median_heuristic(
            subsample_data(X, n_sub_samples, seed),
            subsample_data(Y, n_sub_samples, seed),
        )

        gamma = sigma_to_gamma(sigma)

        # calculate kernel

        return rbf_kernel(X, Y, gamma=gamma)

    return k


def sigma_median_heuristic(X: np.ndarray, Y: np.ndarray) -> np.ndarray:

    dists = squareform(pdist(np.concatenate([X, Y], axis=0), metric="euclidean"))
    median_dist = np.median(dists[dists > 0])
    sigma = median_dist / 2.0  # np.sqrt(2.)

    return sigma


def subsample_data(X, n_samples: int = 1_000, seed: int = 123):

    rng = np.random.RandomState(seed)

    if n_samples < X.shape[0]:

        idx = rng.permutation(n_samples)[:n_samples]

        X = X[idx]

    return X


def sigma_to_gamma(sigma):
    return 1 / (2 * sigma ** 2)


def gamma_to_sigma(gamma):
    return 1 / np.sqrt(2 * gamma)