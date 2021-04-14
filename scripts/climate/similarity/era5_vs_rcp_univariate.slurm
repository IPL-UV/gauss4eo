#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=nodo17,
#SBATCH --workdir=/home/emmanuel/projects/rbig4eo/
#SBATCH --job-name=era5_v_rcp
#SBATCH --output=/home/emmanuel/logs/era5_v_rcp_%A_%a.out


module load Anaconda3
source activate isp_data

PYTHONPATH="/home/emmanuel/projects/rbig4eo/"

srun --ntasks 1 python -u scripts/climate/similarity/era5_vs_rcp.py --experiment univariate --start 201501 --end 202012
