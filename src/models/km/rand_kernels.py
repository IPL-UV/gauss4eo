from typing import Callable
import numpy as np
from sklearn.kernel_approximation import Nystroem, RBFSampler
from src.models.km.kernels import subsample_data, sigma_to_gamma, sigma_median_heuristic


def init_nystrom_proj(
    n_components: int = 100, n_sub_samples: int = 1_000, seed: int = 123, **kwargs
):
    def kernel_proj(X):

        # subsample data
        X_ = subsample_data(X, n_sub_samples, seed)

        # find the sigma via the heuristic
        sigma = sigma_median_heuristic(X_, X_)

        gamma = sigma_to_gamma(sigma)

        # return nystrom kernel
        return Nystroem(
            kernel="rbf", gamma=gamma, n_components=n_components, **kwargs
        ).fit_transform(X)

    return kernel_proj


def init_rff_projection(
    n_components: int = 100, n_sub_samples: int = 1_000, seed: int = 123, **kwargs
):
    def kernel_proj(X):

        # subsample data

        X_ = subsample_data(X, n_sub_samples, seed)

        # find the sigma via the heuristic
        sigma = sigma_median_heuristic(X_, X_)

        gamma = sigma_to_gamma(sigma)

        # calculate projection
        return RBFSampler(
            gamma=gamma, n_components=n_components, **kwargs
        ).fit_transform(X)

    return kernel_proj