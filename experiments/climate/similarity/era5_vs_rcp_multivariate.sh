#!/bin/bash

PYTHONPATH="/home/emmanuel/projects/rbig4eo/"
work_dir=/home/emmanuel/projects/rbig4eo/

models=("cca" "gaussian_mi" "knn_nbs_mi" "knn_eps_mi" "rv_coeff" "nhsic_lin" "nhsic_rbf" "mmd_lin" "mmd_rbf" "mgc")

for model in ${models[@]}; do

    # create job file
    job_file="${work_dir}/multi_${model}.job"

    echo "#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclude=nodo17
#SBATCH --workdir=/home/emmanuel/projects/rbig4eo/
#SBATCH --job-name=multi_${model}
#SBATCH --output=/home/emmanuel/logs/era5_v_rcp_multi.out
#SBATCH --error=/home/emmanuel/errs/era5_v_rcp_multi.err

module load Anaconda3
source activate rbig4eo
PYTHONPATH='/home/emmanuel/projects/rbig4eo/'

srun --ntasks 1 python -u experiments/climate/similarity/era5_vs_rcp_multivariate.py --stats_model ${model} --start 201001 --end 202012" > $job_file

    sbatch $job_file

done