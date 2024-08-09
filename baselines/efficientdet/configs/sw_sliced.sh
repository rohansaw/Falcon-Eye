#!/bin/bash

TRAIN_IMAGE_DIR="/data/sw_sliced/coco/train"
VAL_IMAGE_DIR="/data/sw_sliced/coco/val"
TEST_IMAGE_DIR="/data/sw_sliced/coco/test"
TRAIN_ANNOTATIONS_FILE="/data/sw_sliced/coco/annotations/instances_train-50percent_bg_exp_cleaned.json"
#VAL_ANNOTATIONS_FILE="/data/sw_sliced/coco/annotations/instances_val-50percent_bg_exp_cleaned.json"
VAL_ANNOTATIONS_FILE="/data/sw_sliced/coco/annotations/instances_val.json"
TEST_ANNOTATIONS_FILE="/data/sw_sliced/coco/annotations/instances_test-50percent_bg_exp_cleaned.json"
TFRECORD_DIR="/data/sw_sliced/tfrecord"

TRAINFILE_PATTERN=/data/sw_sliced/tfrecord/train*.tfrecord
VAL_FILE_PATTERN=/data/sw_sliced/tfrecord/val*.tfrecord
TEST_FILE_PATTERN=/data/sw_sliced/tfrecord/test*.tfrecord
HPARAMS_FILE=/configs/hparams/sw_sliced.yaml
MODEL_OUT_DIR=/data/results/efficientdet_lite0_nobn/sw_sliced
NUM_SAMPLES_EVAL=3900
NUM_SAMPLES_EPOCH=13279
BATCH_SIZE=16
EPOCHS=160
