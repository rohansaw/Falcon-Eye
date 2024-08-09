import wandb
from ultralytics import YOLO
from configs import configs
import os
from multiprocessing import Process, Queue
import time
import shutil
from datetime import datetime

project = "yolo"

def train_quantize(data_path, img_sz, batch, model, device, epochs, single_cls, ds_name, workers, cache):
    yolo = YOLO(model)
    yolo.train(data=data_path, epochs=epochs, device=device, batch=batch, workers=workers, imgsz=img_sz,
               single_cls=single_cls, project=project, name=ds_name, cache=cache)

def worker(device, queue):
    while not queue.empty():
        config = queue.get()
        workers = config.get("workers", 8)
        cache = config.get("cache", False)
        if config is not None:
            print(f"Training on GPU: {device} with config: {config['ds_name']}")
            train_quantize(config["data_path"], config["image_size"], config["batch"], config["model"], device, config["epochs"], config["single_cls"], config["ds_name"], workers, cache)

def main():
    devices = [int(dev) for dev in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    queue = Queue()

    for config in configs:
        queue.put(config)

    processes = []
    for device in devices:
        p = Process(target=worker, args=(device, queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    if not os.path.exists("/data/results/yolo/"):
        os.mkdir("/data/results/yolo/")

    now = datetime.now()
    datetime_str = now.strftime("%Y%m%d_%H%M%S")
    shutil.move("yolo/", f"/data/results/yolo/{datetime_str}")

if __name__ == "__main__":
    main()
