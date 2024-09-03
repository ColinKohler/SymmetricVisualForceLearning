#!/bin/bash

main() {
  local env=$1
  local vision_size=$2
  local num_sensors=$3
  local encoder=$4
  local results_path=$5

  for j in $(seq ${6} ${7}); do
    sbatch -J ${env}_cnn_${results_path}_${j} scripts/train_cnn_single_gpu.sbatch $env $vision_size $num_sensors $encoder ${results_path}_cnn_${j}
  done
}

main $1 $2 $3 $4 $5 $6 $7
