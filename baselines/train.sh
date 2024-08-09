#!/bin/bash

# Set the path to the dataset
DATASET_ROOT="/scratch1/rsawahn/data"
WANDB_API_KEY="HIDDEN"

#docker pull rohansaw/search-and-rescue-od:initialexp
#docker run -v $DATASET_ROOT:/data -e WANDB_API_KEY=$WANDB_API_KEY --shm-size 8G --gpus '"device=0"' rohansaw/search-and-rescue-od:initialexp ./train_all.sh > mmdetection_output.log 2> mmdetection_error.log & wait

# docker pull rohansaw/search-and-rescue-od:efficientdet
# docker pull rohansaw/search-and-rescue-od:yolov8
# docker pull rohansaw/search-and-rescue-od:mmdetection

# docker run -v $DATASET_ROOT:/data -e WANDB_API_KEY=$WANDB_API_KEY --shm-size 8G --gpus '"device=0,1,2,3"' rohansaw/search-and-rescue-od:efficientdet ./train_all.sh > efficientdet_output.log 2> efficientdet_error.log &
# docker run -v $DATASET_ROOT:/data -e WANDB_API_KEY=$WANDB_API_KEY --shm-size 8G --gpus '"device=4,5,6,7"' rohansaw/search-and-rescue-od:yolov8 ./train_all.sh > yolov8_output.log 2> yolov8_error.log &
# docker run -v $DATASET_ROOT:/data -e WANDB_API_KEY=$WANDB_API_KEY --shm-size 8G --gpus '"device=8,9,10,11,12,13,14,15"' rohansaw/search-and-rescue-od:mmdetection ./train_all.sh > mmdetection_output.log 2> mmdetection_error.log &


docker run -v $DATASET_ROOT:/data -e WANDB_API_KEY=$WANDB_API_KEY --rm --shm-size 8G --gpus '"device=1"' mmdet ./train_all.sh > out.log 2> err.log & wait