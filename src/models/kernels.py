import sys

pysim_root = "/home/emmanuel/code/pysim"
sys.path.append(str(pysim_root))
from pysim.kernel.hsic import cka_coefficient_rbf
from pysim.linear.rv import rv_coefficient
from pysim.kernel.rhsic import cka_coefficient_rbf_nystroem, cka_coefficient_rff
from pysim.kernel.mmd import mmd_coefficient_rbf
import time


def get_multivariate_kernel_stats(X, Y, subsample=1_000, seed=123):

    stats = {}

    # rv coefficient
    t0 = time.time()
    istat = rv_coefficient(X, Y)
    istat["rv_coeff_time"] = time.time() - t0

    stats = {**stats, **istat}

    # cka coefficient
    t0 = time.time()
    istat = cka_coefficient_rbf(X, Y, subsample=subsample, seed=seed)
    istat["cka_coeff_time"] = time.time() - t0

    stats = {**stats, **istat}

    # cka rff coefficient
    t0 = time.time()
    istat = cka_coefficient_rff(X, Y, subsample=subsample, seed=seed)
    istat["cka_rff_coeff_time"] = time.time() - t0

    stats = {**stats, **istat}

    # cka rff coefficient
    t0 = time.time()
    istat = cka_coefficient_rbf_nystroem(X, Y, subsample=subsample, seed=seed)
    istat["cka_nys_coeff_time"] = time.time() - t0

    stats = {**stats, **istat}

    mmd_coefficient_rbf
    # cka rff coefficient
    t0 = time.time()
    istat = mmd_coefficient_rbf(X, Y, subsample=subsample, seed=seed)
    istat["mmd_coeff_time"] = time.time() - t0

    stats = {**stats, **istat}

    return stats