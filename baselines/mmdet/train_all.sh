#!/bin/bash

AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader)
AVAILABLE_GPUS=($AVAILABLE_GPUS)
echo "${AVAILABLE_GPUS[@]}"
echo "${#AVAILABLE_GPUS[@]}"
CONFIGS_ROOT="/mmdetection/configs/thesis_configs"

wandb login $WANDB_API_KEY

run_with_specific_gpu() {
    local num_gpus=$1
    local job_index=$2
    local config_path=$3
    local gpu_id=$((job_index % num_gpus))
    echo "Running on GPU $gpu_id with config $config_path"
    CUDA_VISIBLE_DEVICES=$gpu_id python3 tools/train.py "$config_path"
}

export -f run_with_specific_gpu

FILES=$(find "$CONFIGS_ROOT" -type f)
IFS=$'\n' read -rd '' -a FILES_ARRAY <<<"$FILES"

num_gpus=${#AVAILABLE_GPUS[@]}
parallel --halt now,fail=1 -j $num_gpus \
    run_with_specific_gpu $num_gpus {#} {} ::: "${FILES_ARRAY[@]}"
