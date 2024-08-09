#!/bin/bash

TRAIN_IMAGE_DIR="/data/sds_sliced_new/coco/train"
VAL_IMAGE_DIR="/data/sds_sliced_new/coco/val"
TRAIN_ANNOTATIONS_FILE="/data/sds_sliced_new/coco/annotations/instances_train-50percent_bg.json"
VAL_ANNOTATIONS_FILE="/data/sds_sliced_new/coco/annotations/instances_val-50percent_bg.json"
TFRECORD_DIR="/data/sds_sliced_new/tfrecord"

TRAINFILE_PATTERN=/data/sds_sliced_new/tfrecord/train*.tfrecord
VAL_FILE_PATTERN=/data/sds_sliced_new/tfrecord/val*.tfrecord
HPARAMS_FILE=/home/thesis/tiny-od-on-edge/baselines/efficientdet/configs/hparams/sds_sliced.yaml
MODEL_OUT_DIR=/data/results/efficientdet-lite0_sds_sliced_new/sds_sliced_new
NUM_SAMPLES_EVAL=19315
NUM_SAMPLES_EPOCH=112783
BATCH_SIZE=16
EPOCHS=40