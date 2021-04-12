from src.models.km.rand_kernels import init_nystrom_proj, init_rff_projection
import numpy as np
from typing import Callable, Dict
import time


def cka_coefficient_nystroem(
    X: np.ndarray,
    Y: np.ndarray,
    n_sub_samples: int = 1_000,
    random_state: int = 123,
) -> Dict:
    """simple function to calculate the rv coefficient"""

    t0 = time.time()

    # estimate kernel
    kern = init_nystrom_proj(n_sub_samples=n_sub_samples, seed=random_state)

    # calculate the kernel matrices
    Kx = kern(X)
    Ky = kern(Y)

    # frobenius norm of the cross terms (numerator)
    xy_norm = rand_v_statistic(Kx, Ky)

    # normalizing coefficients (denomenator)
    x_norm = np.sqrt(rand_v_statistic(Kx, Kx))
    y_norm = np.sqrt(rand_v_statistic(Ky, Ky))

    # rv coefficient
    cka_coeff = xy_norm / x_norm / y_norm

    return {
        "rcka_coeff_nys": cka_coeff,
        "rcka_xy_norm_nys": xy_norm,
        "rcka_x_norm_nys": x_norm,
        "rcka_y_norm_nys": y_norm,
        "rcka_nys_time": time.time() - t0,
    }


def cka_coefficient_rff(
    X: np.ndarray,
    Y: np.ndarray,
    n_sub_samples: int = 1_000,
    random_state: int = 123,
) -> Dict:
    """simple function to calculate the rv coefficient"""

    t0 = time.time()

    # estimate kernel
    kern = init_rff_projection(n_sub_samples=n_sub_samples, seed=random_state)

    # calculate the kernel matrices
    Kx = kern(X)
    Ky = kern(Y)

    # frobenius norm of the cross terms (numerator)
    xy_norm = rand_v_statistic(Kx, Ky)

    # normalizing coefficients (denomenator)
    x_norm = np.sqrt(rand_v_statistic(Kx, Kx))
    y_norm = np.sqrt(rand_v_statistic(Ky, Ky))

    # rv coefficient
    cka_coeff = xy_norm / x_norm / y_norm

    return {
        "rcka_coeff_rff": cka_coeff,
        "rcka_xy_norm_rff": xy_norm,
        "rcka_x_norm_rff": x_norm,
        "rcka_y_norm_rff": y_norm,
        "rcka_time_rff": time.time() - t0,
    }


def rand_v_statistic(phi_x: np.ndarray, phi_y: np.ndarray) -> float:
    """Calculate the biased statistic

    Parameters
    ----------
    phi_x : np.ndarray
        the projection matrix for samples, X
        (n_samples, n_features)

    phi_y : np.ndarray
        the projection matrix for samples, Y
        (n_samples, n_features)

    Returns
    -------
    score : float
        the hsic score using the biased statistic
    """
    n_samples = np.shape(phi_x)[0]

    # remove the mean (from samples)
    phi_x -= np.mean(phi_x, axis=0)
    phi_y -= np.mean(phi_y, axis=0)

    # linear covariance (f x f)
    featCov = phi_x.T @ phi_y

    # normalize
    featCov /= float(n_samples)

    # calculate the norm
    return np.linalg.norm(featCov) ** 2