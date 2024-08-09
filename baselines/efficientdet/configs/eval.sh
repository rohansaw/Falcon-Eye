#!/bin/bash

source "/configs/sw_sliced.sh"

export CUDA_VISIBLE_DEVICES=$GPUS
CUDA_VISIBLE_DEVICES=$GPUS
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

python3 main.py --mode=eval \
    --val_file_pattern=$VAL_FILE_PATTERN \
    --val_json_file=$VAL_ANNOTATIONS_FILE \
    --model_name=efficientdet-lite0 \
    --model_dir=$MODEL_OUT_DIR  \
    --hparams=$HPARAMS_FILE