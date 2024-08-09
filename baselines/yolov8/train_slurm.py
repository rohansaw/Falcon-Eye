import argparse
import json
import wandb
from ultralytics import YOLO
import os
import time
import shutil
from datetime import datetime

project = "yolo"


def train(
    data_path, img_sz, batch, model, device, epochs, single_cls, ds_name, workers, cache
):
    yolo = YOLO(model)
    yolo.train(
        data=data_path,
        epochs=epochs,
        device=device,
        batch=batch,
        workers=workers,
        imgsz=img_sz,
        single_cls=single_cls,
        project=project,
        name=f"{ds_name}_fixedRes",
        cache=cache,
        # translate=0,
        # scale=0,
        # shear=0,
        # mosaic=0,
        # erasing=0,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train a YOLO model with given configuration"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as config_file:
        config = json.load(config_file)

    workers = config.get("workers", 8)
    cache = config.get("cache", False)
    device = "cuda:0"
    train(
        config["data_path"],
        config["image_size"],
        config["batch"],
        config["model"],
        device,
        config["epochs"],
        config["single_cls"],
        config["ds_name"],
        workers,
        cache,
    )

    if not os.path.exists("/data/results/yolo/"):
        os.mkdir("/data/results/yolo/")

    now = datetime.now()
    datetime_str = now.strftime("%Y%m%d_%H%M%S")
    shutil.move("yolo/", f"/data/results/yolo/p2fixed_{datetime_str}")


if __name__ == "__main__":
    main()
