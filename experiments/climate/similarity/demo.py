import sys, os
from pyprojroot import here

# spyder up to find the root
isp_root = "/home/emmanuel/code/isp_data"
root = here(project_files=[".here"])

# append to path
sys.path.append(str(root))
sys.path.append(str(isp_root))

from pathlib import Path

#
import xarray as xr
from pprint import pprint
import cftime
import xesmf as xe
from isp_data.esdc.temporal import convert_to_360day_monthly
from src.models.univariate import pearson
from src.models.kernels import rv_coefficient, cka_coefficient
import tqdm
import numpy as np
import pandas as pd
import itertools

# MATPLOTLIB Settings
import matplotlib as mpl
import matplotlib.pyplot as plt

# SEABORN SETTINGS
import seaborn as sns

sns.set_context(context="talk", font_scale=0.7)

DATASET_DIR = f"{root}/datasets/"
TIME_SLICE = slice("202001", "202012")
BASE_GRID_MODEL = "ipsl_cm5b_lr"


# =====================
# LOAD DATA
# =====================

cmip_glob = Path(DATASET_DIR).joinpath("cmip5").glob(f"*.nc")
era5_glob = Path(DATASET_DIR).joinpath("era5").glob(f"*.nc")
nc_files = [str(x) for x in cmip_glob if x.is_file()] + [
    str(x) for x in era5_glob if x.is_file()
]


ds = {}
for ifile in nc_files:
    # manually set the encoding to use cftime
    i_ds = xr.open_dataset(ifile, decode_times=True, use_cftime=True)

    # assign a new coordinate name
    try:
        model_id = i_ds.model_id.lower().replace("-", "_")

    except:
        model_id = "era5"
        i_ds = i_ds.rename({"msl": "psl", "latitude": "lat", "longitude": "lon"})
    i_ds = i_ds.assign_coords({"model_id": model_id})
    # create the same calendar
    i_ds = convert_to_360day_monthly(i_ds, False)

    # move attributes to new coordinate
    i_ds.model_id.attrs = i_ds.attrs
    i_ds.attrs = {}

    ds[model_id] = i_ds

# =====================
# Select Time Slice
# =====================
for imodel_id, ids in ds.items():
    ds[imodel_id] = ds[imodel_id].sel(time=TIME_SLICE)

n_time = ds[imodel_id]["time"].values.shape[0]

# =====================
# REGRID
# =====================
ds_out = xr.Dataset(
    {
        "lat": (["lat"], ds[BASE_GRID_MODEL].lat),
        "lon": (["lon"], ds[BASE_GRID_MODEL].lon),
    }
)

n_lat = ds_out.coords["lat"].values.shape[0]
n_lon = ds_out.coords["lon"].values.shape[0]

final_ds = []

for imodel_id, ids in ds.items():
    if imodel_id == "ipsl_cm5b_lr":
        t = ids.psl
        t.attrs = ids.attrs
        final_ds.append(t)
        continue

    regridder = xe.Regridder(ids, ds_out, "nearest_s2d")
    t = regridder(ids["psl"])
    t.attrs = ids.attrs
    final_ds.append(t)

final_ds = xr.concat(final_ds, dim="model_id").reset_coords()

# =====================
# PEARSON CORRELATION
# =====================


all_stats = pd.DataFrame()

# get all model IDS
model_ids = final_ds.coords["model_id"].values
sym_pairs = [pair for pair in itertools.combinations(model_ids, 2)]

with tqdm.tqdm(sym_pairs) as pbar:

    for i, imodel_pair in enumerate(pbar):

        imodel_stats = {"model_0": imodel_pair[0], "model_1": imodel_pair[1]}

        pbar.set_description(f"Model 1: {imodel_pair[0]}, Model 2: {imodel_pair[1]}")

        # select model
        ids = final_ds.sel(model_id=[imodel_pair[0], imodel_pair[1]]).psl.values

        # calculate pearson correlation
        pbar.set_description(f"Calculating Correlation...")

        x = np.reshape(ids[0], (n_lat * n_lon * n_time))
        y = np.reshape(ids[1], (n_lat * n_lon * n_time))

        prs_r = pearson(x, y)

        imodel_stats = {**imodel_stats, **prs_r}

        # calculate pearson correlation
        pbar.set_description(f"Calculating RV Coeff...")

        X = np.reshape(ids[0], (n_lat * n_lon, n_time))
        Y = np.reshape(ids[1], (n_lat * n_lon, n_time))
        rv = rv_coefficient(X, Y)

        imodel_stats = {**imodel_stats, **rv}

        # calculate pearson correlation
        pbar.set_description(f"Calculating nHSIC Coeff...")

        nhsic = cka_coefficient(X, Y)

        imodel_stats = {**imodel_stats, **nhsic}

        all_stats = pd.concat(
            [all_stats, pd.DataFrame(imodel_stats, index=[i])], axis=0
        )
