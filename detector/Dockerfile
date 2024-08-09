FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-devel

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

ADD requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt