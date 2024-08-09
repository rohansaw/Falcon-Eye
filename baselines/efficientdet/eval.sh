#!/bin/bash

source /home/thesis/tiny-od-on-edge/baselines/efficientdet/configs/sw_sliced.sh

export CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=0
echo CUDA_VISIBLE_DEVICES=0

mkdir -p testdev_output_sw_sliced_val

MODEL_DIR=/data/results/efficientdet_lite0_nobn/sw_sliced

python3 /data/results/efficientdet_lite0_nobn/sw_sliced/automl2/automl/efficientdet/main.py --mode=eval \
    --val_file_pattern=$VAL_FILE_PATTERN \
    --model_name=efficientdet-lite0 \
    --model_dir=$MODEL_DIR \
    --testdev_dir='testdev_output_sw_sliced_val' \
    --eval_samples=$NUM_SAMPLES_EVAL \
    --hparams=$MODEL_DIR/config.yaml