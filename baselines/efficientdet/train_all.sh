#!/bin/bash

GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader)
GPUS=($GPUS)

#CONFIG_FILES=("/configs/sds.sh" "/configs/sw.sh" "/configs/sds_sliced.sh" "/configs/sw_sliced.sh")
CONFIG_FILES=("/home/thesis/tiny-od-on-edge/baselines/efficientdet/configs/sds_sliced.sh")

for CONFIG_FILE in "${CONFIG_FILES[@]}"
do
    echo "Processing $CONFIG_FILE"

    source $CONFIG_FILE

    export CUDA_VISIBLE_DEVICES=$GPUS
    CUDA_VISIBLE_DEVICES=$GPUS
    echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

    # convert coco dataset to tfrecord if not yet existing
    if ! test -d $TFRECORD_DIR; then
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
    fi

    python3 main.py --mode=train_and_eval \
    --train_file_pattern=$TRAINFILE_PATTERN \
    --val_file_pattern=$VAL_FILE_PATTERN \
    --model_name=efficientdet-lite0 \
    --model_dir=$MODEL_OUT_DIR  \
    --ckpt=efficientdet-lite0 \
    --train_batch_size=$BATCH_SIZE \
    --eval_batch_size=$BATCH_SIZE \
    --eval_samples=$NUM_SAMPLES_EVAL \
    --num_examples_per_epoch=$NUM_SAMPLES_EPOCH --num_epochs=$EPOCHS  \
    --save_checkpoints_steps=$NUM_SAMPLES_EPOCH \
    --hparams=$HPARAMS_FILE

    echo "Finished processing $CONFIG_FILE"
done

echo "Finished processing all configs."