# spyder up to find the root
rbig_root = "/home/emmanuel/code/rbig_jax"
# append to path
import sys

sys.path.append(str(rbig_root))
from src.models.univariate import pearson_dim
from src.models.km.rhsic import cka_coefficient_nystroem, cka_coefficient_rff
from src.models.km.hsic import cka_coefficient
from src.models.km.rv import rv_coefficient

# from rbig_jax.information.mi import rbig_mutual_info
from scipy import stats
import numpy as np

# import jax.numpy as jnp
from typing import Dict
import time


# def multivariate_stats(X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
#     """Calculates some univariate statistics

#     Calculates some standard univeriate statistics such as
#     Pearson, Spearman and KendallTau. Ravels the dataset to
#     ensure that we get a single value instead of one value per
#     feature.

#     Parameters
#     ----------
#     X : np.ndarray, (n_samples, n_features)
#         dataset 1 to be compared

#     Y : np.ndarray, (n_samples, n_features)
#         dataset 2 to be compared

#     Returns
#     -------
#     results : Dict[str, float]
#         a dictionary with the following entries
#         * 'pearson' - pearson correlation coefficient
#         * 'spearman' - spearman correlation coefficient
#         * 'kendalltau' - kendall's tau correlation coefficient
#     """

#     # Reshape
#     results = {}

#     # Pearson Correlation Coefficient
#     prs_r = pearson_dim(X, Y)

#     results = {**results, **prs_r}

#     # linear kernel method
#     rv_ = rv_coefficient(X, Y)

#     results = {**results, **rv_}

#     # linear kernel method
#     nhsic = cka_coefficient(X, Y)

#     results = {**results, **nhsic}

#     # Nystroem
#     rhsic_nys = cka_coefficient_nystroem(X, Y)

#     results = {**results, **rhsic_nys}

#     # RFF
#     rhsic_rff = cka_coefficient_rff(X, Y)

#     results = {**results, **rhsic_rff}

#     # calculate mutual info

#     t0 = time.time()
#     mi_XY_rbig = rbig_mutual_info(
#         X=jnp.array(X, jnp.float32),
#         Y=jnp.array(Y, jnp.float32),
#         zero_tolerance=30,
#     )

#     results["mi_x"] = np.clip(np.array(mi_XY_rbig.mi_X), a_min=0.0, a_max=100)
#     results["mi_y"] = np.array(mi_XY_rbig.mi_Y)
#     results["mi_xy"] = np.array(mi_XY_rbig.mi_XY)
#     results["mi_time"] = time.time() - t0

#     return results
