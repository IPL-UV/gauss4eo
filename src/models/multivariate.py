# spyder up to find the root
rbig_root = "/home/emmanuel/code/rbig_jax"
# append to path
import sys

sys.path.append(str(rbig_root))
from src.models.km.rv import get_rv_coefficient
from src.models.univariate import (
    gaussian_mutual_info,
    knn_mutual_info_nbs,
    knn_mutual_info_eps,
)

# from rbig_jax.information.mi import rbig_mutual_info
from scipy import stats
import numpy as np

# import jax.numpy as jnp
from typing import Dict
import time


def get_multivariate_stats(
    X: np.ndarray, Y: np.ndarray, model: str = "rv", **kwargs
) -> Dict[str, float]:
    """Calculates some univariate statistics

    Calculates some standard univeriate statistics such as
    Pearson, Spearman and KendallTau. Ravels the dataset to
    ensure that we get a single value instead of one value per
    feature.

    Parameters
    ----------
    X : np.ndarray, (n_samples, n_features)
        dataset 1 to be compared

    Y : np.ndarray, (n_samples, n_features)
        dataset 2 to be compared

    model : str
        the model, by default="pearson"

    Returns
    -------
    results : Dict[str, float]
        a dictionary with the following entries
        * 'pearson' - pearson correlation coefficient
        * 'spearman' - spearman correlation coefficient
        * 'kendalltau' - kendall's tau correlation coefficient
    """

    # Reshape
    if X.ndim < 2:
        X = X[:, None]
    if Y.ndim < 2:
        Y = Y[:, None]

    if model == "pearson_dim":
        # Spearman Corr Coeff
        raise NotImplementedError()

    elif model == "cca":

        results = cca_coeff(X, Y)
    elif model == "rv_coeff":
        results = get_rv_coefficient(X, Y)
    elif model == "knn_nbs_mi":
        # K-NN Model
        results = knn_mutual_info_nbs(X, Y, **kwargs)
    elif model == "knn_eps_mi":
        # K-NN Model
        results = knn_mutual_info_eps(X, Y, **kwargs)
    elif model == "gaussian_mi":
        results = gaussian_mutual_info(X, Y)
    elif model == "dcorr":
        results = dcorr_coeff(X, Y)
    elif model == "mgc":
        results = mgc_coeff(X, Y)
    elif model == "energy":
        results = energy_dist(X, Y)
    elif model == "nhsic_lin":
        results = nhsic_coeff(X, Y, kernel="linear")
    elif model == "nhsic_rbf":
        results = nhsic_coeff(X, Y, kernel="rbf")
    elif model == "mmd_lin":
        results = mmd_coeff(X, Y, kernel="linear")
    elif model == "mmd_rbf":
        results = mmd_coeff(X, Y, kernel="rbf")
    elif model == "rbig":
        # RBIG
        raise NotImplementedError()
    else:
        raise ValueError(f"Unrecognized Method: {model}")

    return results


def cca_coeff(
    X: np.ndarray,
    Y: np.ndarray,
) -> Dict[str, float]:
    """Calculates some univariate statistics

    Calculates some standard univeriate statistics such as
    Pearson, Spearman and KendallTau. Ravels the dataset to
    ensure that we get a single value instead of one value per
    feature.

    Parameters
    ----------
    X : np.ndarray, (n_samples, n_features)
        dataset 1 to be compared

    Y : np.ndarray, (n_samples, n_features)
        dataset 2 to be compared

    Returns
    -------
    results : Dict[str, float]
        a dictionary with the following entries
        * 'pearson' - pearson correlation coefficient
        * 'spearman' - spearman correlation coefficient
        * 'kendalltau' - kendall's tau correlation coefficient
    """
    results = {}

    t0 = time.time()
    from hyppo.independence import CCA

    results = {}
    # Pearson Correlation Coefficient
    results["cca"] = CCA().statistic(x=X.copy(), y=Y.copy())
    results["cca_time"] = time.time() - t0

    return results


def dcorr_coeff(
    X: np.ndarray,
    Y: np.ndarray,
) -> Dict[str, float]:
    """Calculates some univariate statistics

    Calculates some standard univeriate statistics such as
    Pearson, Spearman and KendallTau. Ravels the dataset to
    ensure that we get a single value instead of one value per
    feature.

    Parameters
    ----------
    X : np.ndarray, (n_samples, n_features)
        dataset 1 to be compared

    Y : np.ndarray, (n_samples, n_features)
        dataset 2 to be compared

    Returns
    -------
    results : Dict[str, float]
        a dictionary with the following entries
        * 'pearson' - pearson correlation coefficient
        * 'spearman' - spearman correlation coefficient
        * 'kendalltau' - kendall's tau correlation coefficient
    """
    results = {}

    t0 = time.time()
    from hyppo.independence import Dcorr

    results = {}
    # Pearson Correlation Coefficient
    results["dcorr"] = Dcorr(compute_distance="euclidean").statistic(
        x=X.copy(), y=Y.copy()
    )
    results["dcorr_time"] = time.time() - t0

    return results


def energy_dist(
    X: np.ndarray,
    Y: np.ndarray,
) -> Dict[str, float]:
    """Calculates some univariate statistics

    Calculates some standard univeriate statistics such as
    Pearson, Spearman and KendallTau. Ravels the dataset to
    ensure that we get a single value instead of one value per
    feature.

    Parameters
    ----------
    X : np.ndarray, (n_samples, n_features)
        dataset 1 to be compared

    Y : np.ndarray, (n_samples, n_features)
        dataset 2 to be compared

    Returns
    -------
    results : Dict[str, float]
        a dictionary with the following entries
        * 'pearson' - pearson correlation coefficient
        * 'spearman' - spearman correlation coefficient
        * 'kendalltau' - kendall's tau correlation coefficient
    """
    results = {}

    t0 = time.time()
    from hyppo.ksample import Energy

    results = {}
    # Pearson Correlation Coefficient
    results["energy"] = Energy(compute_distance="euclidean").statistic(
        x=X.copy(), y=Y.copy()
    )
    results["energy_time"] = time.time() - t0

    return results


def mgc_coeff(
    X: np.ndarray,
    Y: np.ndarray,
) -> Dict[str, float]:
    """Calculates some univariate statistics

    Calculates some standard univeriate statistics such as
    Pearson, Spearman and KendallTau. Ravels the dataset to
    ensure that we get a single value instead of one value per
    feature.

    Parameters
    ----------
    X : np.ndarray, (n_samples, n_features)
        dataset 1 to be compared

    Y : np.ndarray, (n_samples, n_features)
        dataset 2 to be compared

    Returns
    -------
    results : Dict[str, float]
        a dictionary with the following entries
        * 'pearson' - pearson correlation coefficient
        * 'spearman' - spearman correlation coefficient
        * 'kendalltau' - kendall's tau correlation coefficient
    """
    results = {}

    t0 = time.time()
    from hyppo.independence import MGC

    results = {}
    # Pearson Correlation Coefficient
    results["mgc"] = MGC(compute_distance="euclidean").statistic(x=X.copy(), y=Y.copy())
    results["mgc_time"] = time.time() - t0

    return results


def nhsic_coeff(X: np.ndarray, Y: np.ndarray, kernel: str = "rbf") -> Dict[str, float]:
    """Calculates some univariate statistics

    Calculates some standard univeriate statistics such as
    Pearson, Spearman and KendallTau. Ravels the dataset to
    ensure that we get a single value instead of one value per
    feature.

    Parameters
    ----------
    X : np.ndarray, (n_samples, n_features)
        dataset 1 to be compared

    Y : np.ndarray, (n_samples, n_features)
        dataset 2 to be compared

    Returns
    -------
    results : Dict[str, float]
        a dictionary with the following entries
        * 'pearson' - pearson correlation coefficient
        * 'spearman' - spearman correlation coefficient
        * 'kendalltau' - kendall's tau correlation coefficient
    """
    results = {}

    t0 = time.time()
    from hyppo.independence import Hsic

    results = {}
    # Pearson Correlation Coefficient
    if kernel == "linear":
        results["nhsic_lin"] = Hsic(compute_kernel="linear").statistic(
            x=X.copy(), y=Y.copy()
        )
        results["nhsic_lin_time"] = time.time() - t0
    elif kernel == "rbf":
        results["nhsic_rbf"] = Hsic(compute_kernel="rbf").statistic(
            x=X.copy(), y=Y.copy()
        )
        results["nhsic_rbf_time"] = time.time() - t0
    else:
        raise ValueError(f"Unrecognized kernel: {kernel}")

    return results


def mmd_coeff(X: np.ndarray, Y: np.ndarray, kernel: str = "rbf") -> Dict[str, float]:
    """Calculates some univariate statistics

    Calculates some standard univeriate statistics such as
    Pearson, Spearman and KendallTau. Ravels the dataset to
    ensure that we get a single value instead of one value per
    feature.

    Parameters
    ----------
    X : np.ndarray, (n_samples, n_features)
        dataset 1 to be compared

    Y : np.ndarray, (n_samples, n_features)
        dataset 2 to be compared

    Returns
    -------
    results : Dict[str, float]
        a dictionary with the following entries
        * 'pearson' - pearson correlation coefficient
        * 'spearman' - spearman correlation coefficient
        * 'kendalltau' - kendall's tau correlation coefficient
    """
    results = {}

    t0 = time.time()
    from hyppo.ksample import MMD

    results = {}
    # Pearson Correlation Coefficient
    if kernel == "linear":
        results["mmd_lin"] = MMD(compute_kernel="linear").statistic(
            x=X.copy(), y=Y.copy()
        )
        results["mmd_lin_time"] = time.time() - t0
    elif kernel == "rbf":
        results["mmd_rbf"] = MMD(compute_kernel="rbf").statistic(x=X.copy(), y=Y.copy())
        results["mmd_rbf_time"] = time.time() - t0
    else:
        raise ValueError(f"Unrecognized kernel: {kernel}")

    return results