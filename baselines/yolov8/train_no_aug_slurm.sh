#!/bin/bash -eux
#SBATCH --container-image=/hpi/fs00/home/maximilian.schall/rohansaw+search-and-rescue-od+yolov8.sqsh
#SBATCH -A demelo
#SBATCH --container-workdir=/yolov8
#SBATCH --container-mounts=/hpi/fs00/scratch/maximilian.schall/rohan/data:/data
#SBATCH --container-writable
#SBATCH --partition=sorcery
#SBATCH --cpus-per-task=50
#SBATCH --gpus=1
#SBATCH --mem=80GB
#SBATCH --time=119:00:00
#SBATCH -C x
#SBATCH --array=0
#SBATCH --out logs/%j.txt

CONFIGS_ROOT="/yolov8/run_configs/"

readarray -d '' FILES_ARRAY < <(find "$CONFIGS_ROOT" -type f -print0 | sort -z)

config_path=${FILES_ARRAY[$SLURM_ARRAY_TASK_ID]}

CONVERSION_CONFIG_FILES=("/yolov8/conversion_configs/sds.sh")

for CONVERSION_CONFIG_FILE in "${CONVERSION_CONFIG_FILES[@]}"
do
    source $CONVERSION_CONFIG_FILE
    if ! [ -d "$YOLO_DIR" ]; then
        mkdir $YOLO_DIR
        mkdir $YOLO_DIR/train
        mkdir $YOLO_DIR/val
        ./coco2yolo --annotation-path "$COCO_TRAIN_ANN" --image-download-dir "$COCO_TRAIN_IMG" --category-file "$CONVERSION_INFO_FILE" --task-categories-dir "$YOLO_DIR/train"
        ./coco2yolo --annotation-path "$COCO_VAL_ANN" --image-download-dir "$COCO_VAL_IMG" --category-file "$CONVERSION_INFO_FILE" --task-categories-dir "$YOLO_DIR/val"
    fi
done

export WANDB_API_KEY="PASTE_API_KEY_HERE"
wandb login $WANDB_API_KEY

echo "Running config: $config_path"
echo "Running on GPU: $CUDA_VISIBLE_DEVICES"

CUDA_VISIBLE_DEVICES=0 python3 train_slurm.py --config "$config_path"