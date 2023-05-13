#!/bin/bash
#SBATCH --job-name="HPO_v1"
#SBATCH --output="logs/srun-kerastuner-%j.%N.out"
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=41
#SBATCH --constraint=gpu
#SBATCH --export=ALL
#SBATCH --account=m4331
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sungduk@uci.edu
#SBATCH -t 12:00:00

# ntasks controls the number of keras-tuner workers, which is (ntasks-1).

echo " "
echo "Python Version:"
echo `which python`
echo " "
echo "Loaded modules:"
module list
echo " "

# Keras TUner oracle IP
SYTMP=( $(ip a show dev hsn0 | grep inet | cut -d' ' -f6 | cut -d'/' -f1) )
export NERSC_NODE_HSN_IP=${SYTMP[0]}
echo NERSC_NODE_HSN_IP $NERSC_NODE_HSN_IP

[[ -d ./logs/$2 ]] || mkdir ./logs/$2

# $1: python script name
# $2: Keras Tuner "project name"
srun --mpi=pmi2 --wait=0 bash run-dynamic.gpu-shared.baseline_v1.sh $1 $2
