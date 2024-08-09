#!/bin/bash

TRAIN_IMAGE_DIR="/data/sds/coco/train"
VAL_IMAGE_DIR="/data/sds/coco/val"
TRAIN_ANNOTATIONS_FILE="/data/sds/coco/annotations/instances_train.json"
VAL_ANNOTATIONS_FILE="/data/sds/coco/annotations/instances_val.json"
TFRECORD_DIR="/data/sds/tfrecord"

TRAINFILE_PATTERN=/data/sds/tfrecord/train*.tfrecord
VAL_FILE_PATTERN=/data/sds/tfrecord/val*.tfrecord
HPARAMS_FILE=/configs/hparams/sds.yaml
MODEL_OUT_DIR=/data/results/efficientdet_lite0_noscale_nofreeze/sds
NUM_SAMPLES_EVAL=1547
NUM_SAMPLES_EPOCH=8930
BATCH_SIZE=1
EPOCHS=40