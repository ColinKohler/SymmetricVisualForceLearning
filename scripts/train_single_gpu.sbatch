#!/bin/bash

#SBATCH --job-name=train
#SBATCH --output=scripts/outputs/%j.out
#SBATCH --error=scripts/outputs/%j.err

#SBATCH --nodes=1

#SBATCH --exclude=c[2160,2162-2175,2176-2183],d1008,d1026
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB

#SBATCH --time=04:00:00

#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kohler.c@northeastern.edu

if [ $# -eq 1 ]; then
  python scripts/train.py $1
elif [ $# -eq 2 ]; then
  python scripts/train.py $1 --results_path=$2
else
  python scripts/train.py $1 --results_path=$2 --checkpoint=$3 --buffer=$4
fi
