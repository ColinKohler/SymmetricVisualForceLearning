#!/bin/bash

main() {
  local env=$1
  local vision_size=$2

  for j in $(seq ${3} ${4}); do
    sbatch -J ${env}_${vision_size}_vision_${j} scripts/train_single_gpu.sbatch $env $vision_size 1 vision ${vision_size}_vision_${j}
  done

  for j in $(seq ${3} ${4}); do
    sbatch -J ${env}_${vision_size}_vision_proprio_${j} scripts/train_single_gpu.sbatch $env $vision_size 1 vision+proprio ${vision_size}_vision_proprio_${j}
  done

  for j in $(seq ${3} ${4}); do
    sbatch -J ${env}_${vision_size}_vision_force_${j} scripts/train_single_gpu.sbatch $env $vision_size 1 vision+force ${vision_size}_vision_force_${j}
  done

  for j in $(seq ${3} ${4}); do
    sbatch -J ${env}_${vision_size}_fusion_${j} scripts/train_single_gpu.sbatch $env $vision_size 1 vision+force+proprio ${vision_size}_fusion_${j}
  done
}

main $1 $2 $3 $4 
