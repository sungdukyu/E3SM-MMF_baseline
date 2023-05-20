#!/bin/bash
#SBATCH --job-name="HPO_v1"
#SBATCH --output="logs/srun-kerastuner-%j.%N.out"
#SBATCH --nodes=1
#SBATCH --gpus=v100-16:NUM_GPUS_PER_NODE_HERE
#SBATCH --ntasks=NTASKS_HERE
#SBATCH --export=ALL
#SBATCH --account=atm200007p
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jerryL9@uci.edu
#SBATCH -t 12:00:00


[[ -d ./logs/$2 ]] || mkdir ./logs/$2

# $1: python script name
# $2: Keras Tuner "project name"
source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load anaconda3
conda activate tf2
srun --mpi=pmi2 --wait=0 bash run-dynamic.gpu-shared.baseline_v1.sh $1 $2
