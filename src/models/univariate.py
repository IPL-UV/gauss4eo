from scipy import stats
import numpy as np
from typing import Dict


def univariate_stats(X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
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

    # Pearson Correlation Coefficient
    results["pearson"] = stats.pearsonr(X.ravel(), Y.ravel())[0]

    # Spearman Correlation Coefficient
    results["spearman"] = stats.spearmanr(X.ravel(), Y.ravel())[0]

    # Kendall-Tau Correlation Coefficient
    results["kendall"] = stats.kendalltau(X.ravel(), Y.ravel())[0]

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

    # Pearson Correlation Coefficient
    results["pearson"] = stats.pearsonr(X.ravel(), Y.ravel())[0]

    # Spearman Correlation Coefficient
    results["x_std"] = np.std(X.ravel())

    # Kendall-Tau Correlation Coefficient
    results["y_std"] = np.std(Y.ravel())

    return results
