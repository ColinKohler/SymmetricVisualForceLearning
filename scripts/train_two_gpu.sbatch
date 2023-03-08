#!/bin/bash

#SBATCH --output=scripts/outputs/%j.out
#SBATCH --error=scripts/outputs/%j.err

#SBATCH --nodes=1

#SBATCH --partition=multigpu
#SBATCH --gres=gpu:v100-sxm2:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=60GB
#SBATCH --time=16:00:00

#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kohler.c@northeastern.edu

echo "node: $SLURM_NODELIST"
if [ $# -eq 4 ]; then
  python -u scripts/train.py $1 --vision_size=$2 --num_sensors=$3 --encoder=$4 --num_gpus=2
elif [ $# -eq 5 ]; then
  python -u scripts/train.py $1 --vision_size=$2 --num_sensors=$3 --encoder=$4 --results_path=$5 --num_gpus=2
else
  python -u scripts/train.py $1 --vision_size=$2 --num_sensors=$3 --encoder=$4 --results_path=$5 --checkpoint=$6 --buffer=$7 --num_gpus=2
fi
