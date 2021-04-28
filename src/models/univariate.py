from src.models.km.rhsic import cka_coefficient_nystroem, cka_coefficient_rff
from src.models.km.hsic import cka_coefficient
from scipy import stats
import numpy as np
from typing import Dict
import time


def get_univariate_linear_stats(X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
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

    # Reshape
    results = {}

    x, y = X.ravel(), Y.ravel()

    # Pearson Correlation Coefficient
    prs_r = pearson(x, y)

    results = {**results, **prs_r}

    # Spearman Corr Coeff
    results["spearman"] = stats.spearmanr(x, y)[0]

    # Kendall-Tau Correlation Coefficient
    results["kendall"] = stats.kendalltau(x, y)[0]

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
    results["pearson"] = stats.pearsonr(X.ravel(), Y.ravel())[0]
    results["pearson_time"] = time.time() - t0

    # Spearman Correlation Coefficient
    results["x_std"] = np.std(X.ravel())

    # Kendall-Tau Correlation Coefficient
    results["y_std"] = np.std(Y.ravel())

    return results


def pearson_dim(X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
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

    pearson = []
    x_std = []
    y_std = []

    for ix, iy in zip(X.T, Y.T):
        # Pearson Correlation Coefficient
        pearson.append(stats.pearsonr(ix.ravel(), iy.ravel())[0])
        x_std.append(np.std(ix.ravel()))
        y_std.append(np.std(iy.ravel()))

    results["pearson_d"] = np.mean(pearson)
    results["pearson_dtime"] = time.time() - t0

    # Spearman Correlation Coefficient
    results["x_std_d"] = np.mean(x_std)

    # Kendall-Tau Correlation Coefficient
    results["y_std_d"] = np.mean(y_std)

    return results
