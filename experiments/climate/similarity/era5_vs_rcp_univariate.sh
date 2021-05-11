#!/bin/bash



PYTHONPATH="/home/emmanuel/projects/rbig4eo/"
work_dir=/home/emmanuel/projects/rbig4eo/
output_dir=/home/emmanuel/logs/era5_v_rcp_uni_%A_%a.out
error_dir=/home/emmanuel/errs/era5_v_rcp_uni_%A_%a.err


models=("pearson" "knn_nbs_mi" "knn_eps_mi" "gaussian_mi" "rv_coeff")

for model in ${models[@]}; do

    # create job file
    job_file="${work_dir}/${model}.job"

    echo "#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclude=nodo17
#SBATCH --workdir=/home/emmanuel/projects/rbig4eo/
#SBATCH --job-name=uni_${model}
#SBATCH --output=/home/emmanuel/logs/era5_v_rcp_uni.out
#SBATCH --error=/home/emmanuel/errs/era5_v_rcp_uni.err

module load Anaconda3
source activate rbig4eo
PYTHONPATH='/home/emmanuel/projects/rbig4eo/'

srun --ntasks 1 python -u experiments/climate/similarity/era5_vs_rcp_univariate.py --stats_model ${model} --start 201501 --end 202012" > $job_file

    sbatch $job_file

done