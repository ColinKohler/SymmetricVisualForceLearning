#!/bin/bash

main() {
  local env=$1
  local vision_size=$2
  local encoder=$3
  local n=$4
  local results_path=$5

  for j in $(seq ${6} ${7}); do
    sbatch -J ${env}_${results_path}_${j} scripts/train.sbatch $env $vision_size $encoder $n ${results_path}_${j}
  done
}

main $1 $2 $3 $4 $5 $6 $7
