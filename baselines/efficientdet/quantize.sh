#!/bin/bash

DS_NAME="sw"

MODEL="efficientdet-lite0"
SAVEDMODELDIR_BASE="/data/results/efficientdet"
SAVEDMODELDIR=$SAVEDMODELDIR_BASE/$DS_NAME
FILEPATTERN=/data/$DS_NAME/tfrecord/val*.tfrecord
VAl_JSON=/data/$DS_NAME/coco/annotations/instances_val.json
HPARAMS=/configs/hparams/$DS_NAME.yaml
mkdir -p $SAVEDMODELDIR

python3 -m tf2.inspector --mode=export --file_pattern=/data/sw/tfrecord/val*.tfrecord \
  --model_name=$MODEL --model_dir=$SAVEDMODELDIR --num_calibration_steps=100 \
  --saved_model_dir=$SAVEDMODELDIR --use_xla --tflite=INT8 --hparams=$HPARAMS

# python3 -m tf2.eval_tflite  \
#     --model_name=$MODEL  --tflite_path=$SAVEDMODELDIR/int8.tflite   \
#     --val_file_pattern=$FILEPATTERN \
#     --val_json_file=$VAl_JSON --eval_samples=500 --hparams=$HPARAMS