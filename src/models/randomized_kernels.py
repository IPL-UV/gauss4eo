from typing import Callable
import numpy as np
from sklearn.kernel_approximation import Nystroem, RBFSampler


class randHSIC:
    def __init__(
        self,
        kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ):
        self.kernel = kernel

    def score(self, X, Y, normalized: bool = True):

        # calculate kernel matrices
        Zx = self.kernel(X)
        Zy = self.kernel(Y)

        # calculate hsic score
        hsic_score = rand_v_statistic(Zx, Zy)

        # normalized
        if normalized:
            norm = rand_v_statistic(Zx, Zx) * rand_v_statistic(Zy, Zy)

            hsic_score /= np.sqrt(norm)

        # numerical error
        rhsic_score = np.clip(hsic_score, a_min=0.0, a_max=100_000)

        return rhsic_score


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


def init_nystrom_proj(**kwargs):
    def kernel_proj(X):
        return Nystroem(**kwargs).fit_transform(X)

    return kernel_proj


def init_rff_projection(n_sub_samples: int = 1_000, seed: int = 123, **kwargs):
    def k(X):

        # subsample data

        X_ = subsample_data(X, n_sub_samples, seed)

        sigma = sigma_median_heuristic(X_, X_)

        gamma = sigma_to_gamma(sigma)

        # calculate projection
        return RBFSampler(gamma=gamma, **kwargs).fit_transform(X)

    return k