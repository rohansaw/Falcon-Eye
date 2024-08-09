#!/bin/bash

source configs/sds.sh

export CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=0
echo CUDA_VISIBLE_DEVICES=0

python3 /data/results/efficientdet_lite0_nobn/sw_sliced/automl2/automl/efficientdet/model_inspect.py --runmode=bm --model_name=efficientdet-lite0 \
  --ckpt_path=$MODEL_OUT_PATH \
  --batch_size=1 \
  --bm_runs=100 \
  --hparams=/configs/hparams/sds.yaml