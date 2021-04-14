from src.models.multivariate import multivariate_stats
import sys, os
from pyprojroot import here

# spyder up to find the root
isp_root = "/home/emmanuel/code/isp_data"
root = here(project_files=[".here"])

# append to path
sys.path.append(str(root))
sys.path.append(str(isp_root))

from pathlib import Path
from argparse import ArgumentParser

#
import xarray as xr
import xesmf as xe
from isp_data.esdc.temporal import convert_to_360day_monthly
from src.models.univariate import univariate_stats
import tqdm
import numpy as np
import pandas as pd
import itertools

import wandb

# MATPLOTLIB Settings
import matplotlib as mpl
import matplotlib.pyplot as plt

# SEABORN SETTINGS
import seaborn as sns

sns.set_context(context="talk", font_scale=0.7)


ERA5_DIR = "/media/disk/databases/CLIMATE/CLIMATE_DATA_STORE/STAGING/SIM4CLIM/ERA5/REANALYSIS/ZARR"
CMIP5_DIR = "/media/disk/databases/CLIMATE/CLIMATE_DATA_STORE/STAGING/SIM4CLIM/CMIP5/RCP8_5/NETCDF"
VARIABLE = "mean_sea_level_pressure"

# ==========================
# ARGS
# ==========================

parser = ArgumentParser(
    description="2D Data Demo with Iterative Gaussianization method"
)

# ==========================
# PARAMETERS
# ==========================
parser.add_argument("--start", type=int, default=202001)
parser.add_argument("--end", type=int, default=202012)
parser.add_argument("--base_grid", type=str, default="ipsl_cm5b_lr")

# ======================
# Logger Parameters
# ======================
parser.add_argument("--name", type=str, default="univariate")
parser.add_argument("--experiment", type=str, default="univariate")
parser.add_argument("--wandb-entity", type=str, default="ipl_uv")
parser.add_argument("--wandb-project", type=str, default="sim4clim_era5vcmip5")


# =====================
# Testing
# =====================
parser.add_argument(
    "-sm",
    "--smoke-test",
    action="store_true",
    help="to do a smoke test without logging",
)

# =====================
# PARSE
# =====================
args = parser.parse_args()

# ==========================
# INITIALIZE LOGGER
# ==========================
if args.smoke_test:
    os.environ["WANDB_MODE"] = "dryrun"

wandb_logger = wandb.init(project=args.wandb_project, entity=args.wandb_entity)
wandb_logger.config.update(args)
config = wandb_logger.config

TIME_SLICE = slice(str(config.start), str(config.end))

cmip_model_names = {
    "access1_0": "ACCESS-1.0",
    "access1_3": "ACCESS-3.0",
    "bnu_esm": "BNU-ESM",
    "era5": "ERA5",
    "inmcm4": "INMCM4",
    "ipsl_cm5a_lr": "IPSL-CM5A-LR",
    "ipsl_cm5b_lr": "IPSL-CM5B-LR",
    "ipsl_cm5a_mr": "IPSL-CM5A-MR",
    "mpi_esm_lr": "MPI-ESM-LR",
    "mpi_esm_mr": "MPI-ESM-MR",
    "giss_e2_h_cc": "GISS-E2-H-CC",
    "fio_esm": "FIO-ESM",
    "noresm1_m": "NORESM1-M",
    "ccsm4": "CCSM4",
}
stats_model_names = {
    "pearson": "Pearson Coeff",
    "pearson_d": "Pearson Coeff (Dim)",
    "spearman": "Spearman Coeff",
    "kendall": "Kendall-Tau Coeff",
    "rv_coeff": "RV Coeff",
    "cka_coeff": "nHSIC",
    "rcka_coeff_nys": "nHSIC (Nystroem)",
    "rcka_coeff_rff": "nHSIC (RFF)",
    "mi_xy": "MI (RBIG)",
}
# =====================
# LOAD DATA
# =====================

cmip_glob = Path(CMIP5_DIR).joinpath(VARIABLE).glob(f"*.nc")
era5_glob = Path(ERA5_DIR).joinpath(VARIABLE).glob(f"*.nc")
nc_files = [str(x) for x in cmip_glob if x.is_dir()]
nc_files = nc_files + [str(x) for x in era5_glob if x.is_dir()]

# SELECT THE VARIABLE

ds = {}
for ifile in nc_files:
    # manually set the encoding to use cftime

    i_ds = xr.open_zarr(ifile, decode_times=True, use_cftime=True)

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
with tqdm.tqdm(ds.items()) as pbar:
    for imodel_id, ids in pbar:

        pbar.set_description(f"Model: {model_id}")

        ds[imodel_id] = ds[imodel_id].sel(time=TIME_SLICE)

n_time = ds[imodel_id]["time"].values.shape[0]
wandb.log({"n_time": n_time})
# =====================
# REGRID
# =====================
ds_out = xr.Dataset(
    {
        "lat": (["lat"], ds[args.base_grid].lat),
        "lon": (["lon"], ds[args.base_grid].lon),
    }
)

n_lat = ds_out.coords["lat"].values.shape[0]
n_lon = ds_out.coords["lon"].values.shape[0]
wandb.log({"n_lat": n_lat})
wandb.log({"n_lon": n_lon})

final_ds = []

for imodel_id, ids in ds.items():
    #     if imodel_id == "ipsl_cm5b_lr":
    #         t = ids.psl
    #         t.attrs = ids.attrs
    #         final_ds.append(t)
    #         continue

    regridder = xe.Regridder(ids, ds_out, "nearest_s2d")
    t = regridder(ids["psl"].compute())
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

        imodel_stats = {
            "model_0": cmip_model_names[imodel_pair[0]],
            "model_1": cmip_model_names[imodel_pair[1]],
        }

        pbar.set_description(f"Model 1: {imodel_pair[0]}, Model 2: {imodel_pair[1]}")

        # select model
        ids = final_ds.sel(model_id=[imodel_pair[0], imodel_pair[1]]).psl.values

        if args.experiment == "univariate":

            # calculate pearson correlation
            # pbar.set_description(f"Calculating Univariate Stats...")

            x = np.reshape(ids[0], (n_lat * n_lon * n_time))
            y = np.reshape(ids[1], (n_lat * n_lon * n_time))

            uni_stats = univariate_stats(x, y)

            imodel_stats = {**imodel_stats, **uni_stats}

        elif args.experiment == "multivariate":

            # calculate pearson correlation
            # pbar.set_description(f"Calculating multivariate Stats...")

            x = np.reshape(ids[0], (n_lat * n_lon, n_time))
            y = np.reshape(ids[1], (n_lat * n_lon, n_time))

            multi_stats = multivariate_stats(x, y)

            imodel_stats = {**imodel_stats, **multi_stats}
        else:
            raise ValueError(f"Unrecognized experiment {args.experiment}")

        all_stats = pd.concat(
            [all_stats, pd.DataFrame(imodel_stats, index=[i])], axis=0
        )
        wandb.log({f"stats": wandb.Table(dataframe=all_stats)})
        if args.smoke_test:
            break


if args.experiment == "univariate":

    # calculate pearson correlation
    stats_model = [
        "pearson",
        "spearman",
        "kendall",
    ]

elif args.experiment == "multivariate":

    stats_model = [
        "pearson_d",
        "rv_coeff",
        "cka_coeff",
        "rcka_coeff_nys",
        "rcka_coeff_rff",
        "mi_xy",
    ]
else:
    raise ValueError(f"Unrecognized experiment {args.experiment}")


from src.visualization.climate.utils import get_pivot_df, plot_dendrogram, plot_heatmap


plt_kwargs = {"vmin": 0, "cmap": "Reds"}

for istats_model in stats_model:

    # create pivot table
    df = get_pivot_df(all_stats, istats_model)

    # plot heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    ax = plot_heatmap(df, ax, **plt_kwargs)
    ax.set(title=stats_model_names[istats_model])
    plt.tight_layout()
    wandb.log({f"heat_map_{istats_model}": wandb.Image(plt)})

    # plot dendrogram
    fig, ax = plt.subplots(figsize=(6, 5))
    ax = plot_dendrogram(df, ax, cmip_model_names)
    ax.set(title=stats_model_names[istats_model])
    plt.tight_layout()
    wandb.log({f"dendrogram_{istats_model}": wandb.Image(plt)})
