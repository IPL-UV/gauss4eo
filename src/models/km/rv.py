import numpy as np
from typing import Callable, Dict, Optional
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import KernelCenterer
from src.models.km.hsic import hsic_v_statistic
import time


def rv_coefficient(
    X: np.ndarray,
    Y: np.ndarray,
) -> Dict:
    """simple function to calculate the rv coefficient"""
    t0 = time.time()
    # calculate the kernel matrices
    Kx = linear_kernel(X)
    Ky = linear_kernel(Y)

    # frobenius norm of the cross terms (numerator)
    xy_norm = hsic_v_statistic(Kx, Ky)

    # normalizing coefficients (denomenator)
    x_norm = np.sqrt(hsic_v_statistic(Kx, Kx))
    y_norm = np.sqrt(hsic_v_statistic(Ky, Ky))

    # rv coefficient
    rv_coeff = xy_norm / x_norm / y_norm

    return {
        "rv_coeff": rv_coeff,
        "rv_xy_norm": xy_norm,
        "rv_x_norm": x_norm,
        "rv_y_norm": y_norm,
        "rv_time": time.time() - t0,
    }