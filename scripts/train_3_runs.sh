#!/bin/bash

main() {
  local env=$1
  local num_sensors=$2
  local encoder=$3
  local results_path=$4

  for j in {1..3}; do
    sbatch scripts/train_single_gpu.sbatch $env $num_sensors $encoder ${results_path}_${j}
  done
}

main $1 $2 $3 $4
