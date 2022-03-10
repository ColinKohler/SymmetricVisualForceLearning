#!/bin/bash

train() {
  local j=$1
  local env=$2
  local checkpoint_name=$3
  local job_id=$4

  if [ $j -eq 1 ]; then
    ret_val=$(sbatch scripts/train_single_gpu.sbatch $env $checkpoint_name)
  else
    ret_val=$(sbatch --dependency=afterany:$job_id scripts/train_single_gpu.sbatch $env $checkpoint_name $checkpoint_name $checkpoint_name)
  fi

  echo $ret_val
  ret_val=${ret_val##* }
}


main() {
  local env=$1
  local checkpoint_name=$2
  local ret_val

  for j in {1..15}; do
    if [ $j -eq 1 ]; then
      train $j $env $checkpoint_name $ret_val
    else
      train $j $env $checkpoint_name $ret_val
    fi
  done
}

main $1 $2
