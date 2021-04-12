from typing import Callable

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import KernelCenterer
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from typing import Dict, Optional
import time
from scipy.spatial.distance import squareform, pdist, cdist


def rv_coefficient(
    X: np.ndarray,
    Y: np.ndarray,
    subsample: Optional[int] = 10_000,
    random_state: int = 123,
) -> Dict:
    """simple function to calculate the rv coefficient"""

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


class HSIC:
    def __init__(
        self,
        kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ):
        self.kernel = kernel

    def score_u_stat(self, X, Y, normalized: bool = True):

        # calculate kernel matrices
        Kx = self.kernel(X, X)
        Ky = self.kernel(Y, Y)

        # calculate hsic score
        hsic_score = hsic_u_statistic(Kx, Ky)

        # normalized
        if normalized:
            norm = hsic_u_statistic(Kx, Kx) * hsic_u_statistic(Ky, Ky)

            hsic_score /= np.sqrt(norm)

        # numerical error
        hsic_score = np.clip(hsic_score, a_min=0.0, a_max=hsic_score)

        return hsic_score

    def score_v_stat(self, X, Y, normalized: bool = True):

        # calculate kernel matrices
        Kx = self.kernel(X, X)
        Ky = self.kernel(Y, Y)

        # calculate hsic score
        hsic_score = hsic_v_statistic(Kx, Ky)

        # normalized
        if normalized:
            norm = hsic_v_statistic(Kx, Kx) * hsic_v_statistic(Ky, Ky)

            hsic_score /= np.sqrt(norm)

        # numerical error
        hsic_score = np.clip(hsic_score, a_min=0.0, a_max=hsic_score)

        return hsic_score


def hsic_u_statistic(K_x: np.ndarray, K_y: np.ndarray) -> float:
    """Calculate the unbiased statistic

    Parameters
    ----------
    K_x : np.ndarray
        the kernel matrix for samples, X
        (n_samples, n_samples)

    K_y : np.ndarray
        the kernel matrix for samples, Y

    Returns
    -------
    score : float
        the hsic score using the unbiased statistic
    """
    n_samples = K_x.shape[0]

    np.fill_diagonal(K_x, 0.0)
    np.fill_diagonal(K_y, 0.0)

    K_xy = K_x @ K_y

    # Term 1
    a = 1 / n_samples / (n_samples - 3)
    A = np.trace(K_xy)

    # Term 2
    b = a / (n_samples - 1) / (n_samples - 2)
    B = np.sum(K_x) * np.sum(K_y)

    # Term 3
    c = (a * 2) / (n_samples - 2)
    C = np.sum(K_xy)

    # calculate hsic statistic
    return a * A + b * B - c * C


def hsic_v_statistic(K_x: np.ndarray, K_y: np.ndarray) -> float:
    """Calculate the biased statistic

    Parameters
    ----------
    K_x : np.ndarray
        the kernel matrix for samples, X
        (n_samples, n_samples)

    K_y : np.ndarray
        the kernel matrix for samples, Y

    Returns
    -------
    score : float
        the hsic score using the biased statistic
    """
    n_samples = K_x.shape[0]

    # center the kernel matrices
    K_x = KernelCenterer().fit_transform(K_x)
    K_y = KernelCenterer().fit_transform(K_y)

    # calculate hsic statistic
    return float(np.einsum("ij,ij->", K_x, K_y) / n_samples ** 2)


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