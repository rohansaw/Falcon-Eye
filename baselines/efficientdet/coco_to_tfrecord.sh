#!/bin/bash

CONFIG_FILES=("/configs/sds.sh" "/configs/sw.sh" "/configs/sds_sliced.sh" "/configs/sw_sliced.sh")

for CONFIG_FILE in "${CONFIG_FILES[@]}"
do
    echo "Processing $CONFIG_FILE"

    source $CONFIG_FILE
    PYTHONPATH=".:$PYTHONPATH" python3 dataset/create_coco_tfrecord.py \
        --image_dir="${TRAIN_IMAGE_DIR}" \
        --object_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
        --output_file_prefix="${TFRECORD_DIR}/train" \
        --num_shards=32

    PYTHONPATH=".:$PYTHONPATH" python3 dataset/create_coco_tfrecord.py \
        --image_dir="${VAL_IMAGE_DIR}" \
        --object_annotations_file="${VAL_ANNOTATIONS_FILE}" \
        --output_file_prefix="${TFRECORD_DIR}/val" \
        --num_shards=32
done