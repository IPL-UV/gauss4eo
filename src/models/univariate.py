# from src.models.km.rhsic import cka_coefficient_nystroem, cka_coefficient_rff
# from src.models.km.hsic import cka_coefficient
import sys, os

pysim_root = "/home/emmanuel/code/pysim"
sys.path.append(str(pysim_root))
from scipy import stats
import numpy as np
from typing import Dict
import time

from src.models.km.rv import get_rv_coefficient
from npeet.entropy_estimators import mi as npeet_mutual_info
from sklearn.preprocessing import StandardScaler
from pysim.information.gaussian import gauss_entropy_multi
from pysim.information.mutual import multivariate_mutual_information


def get_univariate_stats(
    X: np.ndarray, Y: np.ndarray, model: str = "pearson", **kwargs
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

    if model == "pearson":
        # Spearman Corr Coeff
        results = pearson(X.ravel(), Y.ravel())
    elif model == "knn_nbs_mi":
        # K-NN Model
        results = knn_mutual_info_nbs(X, Y, **kwargs)
    elif model == "knn_eps_mi":
        # K-NN Model
        results = knn_mutual_info_eps(X, Y, **kwargs)
    elif model == "rv_coeff":
        results = get_rv_coefficient(X, Y)
    elif model == "gaussian_mi":
        results = gaussian_mutual_info(X, Y)
    elif model == "rbig":
        # RBIG
        raise NotImplementedError()
    else:
        raise ValueError(f"Unrecognized Method: {model}")

    return results


def pearson(X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
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

    # Pearson Correlation Coefficient
    results["pearson"] = stats.pearsonr(X, Y)[0]
    results["pearson_time"] = time.time() - t0

    # Spearman Correlation Coefficient
    results["pearsn_x_std"] = np.std(X)

    # Kendall-Tau Correlation Coefficient
    results["pearson_y_std"] = np.std(Y)

    return results


def knn_mutual_info_nbs(
    X: np.ndarray,
    Y: np.ndarray,
    n_neighbors: int = 10,
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
    from mutual_info.mutual_info import mutual_information as mi_mutual_info

    if X.ndim < 2:
        X = X[:, None]
    if Y.ndim < 2:
        Y = Y[:, None]

    # Pearson Correlation Coefficient
    results["knn_nbs_mi"] = mi_mutual_info(
        (X.copy(), Y.copy()), k=n_neighbors, transform=None
    )
    results["knn_nbs_time"] = time.time() - t0

    return results


def knn_mutual_info_eps(
    X: np.ndarray,
    Y: np.ndarray,
    n_neighbors: int = 10,
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
    from mutual_info.mutual_info import mutual_information as mi_mutual_info

    if X.ndim < 2:
        X = X[:, None]
    if Y.ndim < 2:
        Y = Y[:, None]

    # Pearson Correlation Coefficient
    results["knn_eps_mi"] = npeet_mutual_info(
        X.copy(), Y.copy(), k=n_neighbors, base=np.e
    )
    results["knn_eps_time"] = time.time() - t0

    return results


def gaussian_mutual_info(X, Y):

    if X.ndim < 2:
        X = X[:, None]
    if Y.ndim < 2:
        Y = Y[:, None]

    results = {}
    t0 = time.time()
    stats = multivariate_mutual_information(
        X=X.copy(), Y=Y.copy(), f=gauss_entropy_multi
    )
    results["gaussian_mi_time"] = time.time() - t0
    results["gaussian_mi"] = stats.MI
    results["gaussian_H_XY"] = stats.H_XY
    results["gaussian_H_X"] = stats.H_X
    results["gaussian_H_Y"] = stats.H_Y
    return results
