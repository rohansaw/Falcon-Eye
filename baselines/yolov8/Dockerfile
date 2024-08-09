FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt update
RUN apt install git htop nano -y 
RUN apt install wget -y 
RUN apt install python3 python3-pip -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install ultralytics wandb
RUN pip install pycocotools requests click sahi
ADD . /yolov8
RUN chmod +x /yolov8/train_all.sh
RUN chmod +x /yolov8/coco2yolo
WORKDIR /yolov8