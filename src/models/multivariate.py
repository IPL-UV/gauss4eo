from src.models.univariate import pearson_dim
from src.models.km.rhsic import cka_coefficient_nystroem, cka_coefficient_rff
from src.models.km.hsic import cka_coefficient
from src.models.km.rv import rv_coefficient
from scipy import stats
import numpy as np
from typing import Dict
import time


def multivariate_stats(X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
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

    # Pearson Correlation Coefficient
    prs_r = pearson_dim(X, Y)

    results = {**results, **prs_r}

    # linear kernel method
    rv_ = rv_coefficient(X, Y)

    results = {**results, **rv_}

    # linear kernel method
    nhsic = cka_coefficient(X, Y)

    results = {**results, **nhsic}

    # Nystroem
    rhsic_nys = cka_coefficient_nystroem(X, Y)

    results = {**results, **rhsic_nys}

    # RFF
    rhsic_rff = cka_coefficient_rff(X, Y)

    results = {**results, **rhsic_rff}

    return results
