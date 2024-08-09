#!/bin/bash

GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader)
GPUS=($GPUS)

export CUDA_VISIBLE_DEVICES=$GPUS
CUDA_VISIBLE_DEVICES=$GPUS

CONFIG_FILES=("/yolov8/conversion_configs/sds.sh" "/yolov8/conversion_configs/sw.sh" "/yolov8/conversion_configs/sds_sliced.sh" "/yolov8/conversion_configs/sw_sliced.sh")

for CONFIG_FILE in "${CONFIG_FILES[@]}"
do
    source $CONFIG_FILE
    if ! [ -d "$YOLO_DIR" ]; then
        mkdir $YOLO_DIR
        mkdir $YOLO_DIR/train
        mkdir $YOLO_DIR/val
        ./coco2yolo --annotation-path "$COCO_TRAIN_ANN" --image-download-dir "$COCO_TRAIN_IMG" --category-file "$CONVERSION_INFO_FILE" --task-categories-dir "$YOLO_DIR/train"
        ./coco2yolo --annotation-path "$COCO_VAL_ANN" --image-download-dir "$COCO_VAL_IMG" --category-file "$CONVERSION_INFO_FILE" --task-categories-dir "$YOLO_DIR/val"
    fi
done

wandb login $WANDB_API_KEY

python3 train.py