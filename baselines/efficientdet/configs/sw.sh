#!/bin/bash

TRAIN_IMAGE_DIR="/data/sw/coco/train"
VAL_IMAGE_DIR="/data/sw/coco/val"
TEST_IMAGE_DIR="/data/sw/coco/test"
TRAIN_ANNOTATIONS_FILE="/data/sw/coco/annotations/instances_train_exp_cleaned.json"
VAL_ANNOTATIONS_FILE="/data/sw/coco/annotations/instances_val_exp_cleaned.json"
TEST_ANNOTATIONS_FILE="/data/sw/coco/annotations/instances_test_exp_cleaned.json"
TFRECORD_DIR="/data/sw/tfrecord"

TRAINFILE_PATTERN=/data/sw/tfrecord/train*.tfrecord
VAL_FILE_PATTERN=/data/sw/tfrecord/val*.tfrecord
TEST_FILE_PATTERN=/data/sw/tfrecord/test*.tfrecord
HPARAMS_FILE=/configs/hparams/sw.yaml
MODEL_OUT_DIR=/data/results/efficientdet_lite0_nobn/sw_cleaned
NUM_SAMPLES_EVAL=1619
NUM_SAMPLES_EPOCH=3634
BATCH_SIZE=1
EPOCHS=80