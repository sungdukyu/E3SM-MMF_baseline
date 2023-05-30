#!/bin/bash
#SBATCH --job-name="HPO_v1"
#SBATCH --output="logs/srun-kerastuner-%j.%N.out"
#SBATCH --nodes=1
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-16:4
#SBATCH --ntasks=5
#SBATCH --export=ALL
#SBATCH --account=atm200007p
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jerryL9@uci.edu
#SBATCH -t 0:25:00


[[ -d ./logs/bair ]] || mkdir ./logs/bair

# $1: python script name
# $2: Keras Tuner "project name"
source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load anaconda3
conda activate bair
srun --mpi=pmi2 --wait=0 bash run-dynamic.gpu-shared.baseline_v1.sh $1
