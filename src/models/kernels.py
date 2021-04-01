from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import KernelCenterer
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from typing import Dict, Optional
import time


def rv_coefficient(
    X: np.ndarray,
    Y: np.ndarray,
    subsample: Optional[int] = 10_000,
    random_state: int = 123,
) -> Dict:
    """simple function to calculate the rv coefficient"""
    t0 = time.time()

    # calculate the kernel matrices
    X_gram = linear_kernel(X)
    Y_gram = linear_kernel(Y)

    # center the kernels
    X_gramc = KernelCenterer().fit_transform(X_gram)
    Y_gramc = KernelCenterer().fit_transform(Y_gram)

    # normalizing coefficients (denomenator)
    x_norm = np.linalg.norm(X_gramc)
    y_norm = np.linalg.norm(Y_gramc)

    # frobenius norm of the cross terms (numerator)
    xy_norm = np.sum(X_gramc * Y_gramc)
    # rv coefficient
    pv_coeff = xy_norm / x_norm / y_norm

    return {
        "rv_coef": pv_coeff,
        "x_norm": x_norm,
        "y_norm": y_norm,
        "xy_norm": xy_norm,
    }


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


def cka_coefficient(
    X: np.ndarray,
    Y: np.ndarray,
    random_state: int = 123,
) -> Dict:
    """simple function to calculate the rv coefficient"""
    # estimate sigmas
    sigma_X = estimate_sigma(X, percent=50)
    sigma_Y = estimate_sigma(Y, percent=50)

    # calculate the kernel matrices
    X_gram = RBF(sigma_X)(X)
    Y_gram = RBF(sigma_Y)(Y)

    # center the kernels
    X_gram = KernelCenterer().fit_transform(X_gram)
    Y_gram = KernelCenterer().fit_transform(Y_gram)

    # normalizing coefficients (denomenator)
    x_norm = np.linalg.norm(X_gram)
    y_norm = np.linalg.norm(Y_gram)

    # frobenius norm of the cross terms (numerator)
    xy_norm = np.sum(X_gram * Y_gram)
    # rv coefficient
    pv_coeff = xy_norm / x_norm / y_norm

    return {
        "cka_coeff": pv_coeff,
        "cka_y_norm": y_norm,
        "cka_x_norm": x_norm,
        "cka_xy_norm": xy_norm,
    }