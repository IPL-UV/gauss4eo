from src.models.km.kernels import init_rbf_kernel
from typing import Callable, Dict
import numpy as np
from sklearn.preprocessing import KernelCenterer
import time


def cka_coefficient(
    X: np.ndarray,
    Y: np.ndarray,
    n_sub_samples: int = 1_000,
    random_state: int = 123,
) -> Dict:
    """simple function to calculate the rv coefficient"""
    t0 = time.time()
    # estimate kernel
    kern = init_rbf_kernel(n_sub_samples=n_sub_samples, seed=random_state)

    # calculate the kernel matrices
    Kx = kern(X, X)
    Ky = kern(Y, Y)

    # frobenius norm of the cross terms (numerator)
    xy_norm = hsic_v_statistic(Kx, Ky)

    # normalizing coefficients (denomenator)
    x_norm = np.sqrt(hsic_v_statistic(Kx, Kx))
    y_norm = np.sqrt(hsic_v_statistic(Ky, Ky))

    # rv coefficient
    cka_coeff = xy_norm / x_norm / y_norm

    return {
        "cka_coeff": cka_coeff,
        "cka_xy_norm": xy_norm,
        "cka_x_norm": x_norm,
        "cka_y_norm": y_norm,
        "cka_time": time.time() - t0,
    }


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
