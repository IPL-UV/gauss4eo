#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=nodo17,
#SBATCH --workdir=/home/emmanuel/projects/rbig4eo/
#SBATCH --job-name=era5_v_rcp_m
#SBATCH --output=/home/emmanuel/logs/era5_v_rcp_m_%A_%a.out


module load Anaconda3
source activate isp_data

PYTHONPATH="/home/emmanuel/projects/rbig4eo/"

srun --ntasks 1 python -u scripts/climate/similarity/era5_vs_rcp.py --experiment multivariate --start 201001 --end 201012
srun --ntasks 1 python -u scripts/climate/similarity/era5_vs_rcp.py --experiment multivariate --start 201101 --end 201112
srun --ntasks 1 python -u scripts/climate/similarity/era5_vs_rcp.py --experiment multivariate --start 201301 --end 201312
srun --ntasks 1 python -u scripts/climate/similarity/era5_vs_rcp.py --experiment multivariate --start 201401 --end 201412
srun --ntasks 1 python -u scripts/climate/similarity/era5_vs_rcp.py --experiment multivariate --start 201501 --end 201512
srun --ntasks 1 python -u scripts/climate/similarity/era5_vs_rcp.py --experiment multivariate --start 201601 --end 201612
srun --ntasks 1 python -u scripts/climate/similarity/era5_vs_rcp.py --experiment multivariate --start 201701 --end 201712
srun --ntasks 1 python -u scripts/climate/similarity/era5_vs_rcp.py --experiment multivariate --start 201801 --end 201812
srun --ntasks 1 python -u scripts/climate/similarity/era5_vs_rcp.py --experiment multivariate --start 201901 --end 201912
srun --ntasks 1 python -u scripts/climate/similarity/era5_vs_rcp.py --experiment multivariate --start 202001 --end 202012
srun --ntasks 1 python -u scripts/climate/similarity/era5_vs_rcp.py --experiment multivariate --start 201001 --end 202012
