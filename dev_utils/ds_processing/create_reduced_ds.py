import json
import os
import random

coco_path = ""

new_images = []
new_annotations = []

with open(coco_path, "r") as f:
    data = json.load(f)

    # randomly sample 500 images that contain annotations

    imgs_with_anns = {ann["image_id"]: True for ann in data["annotations"]}
    img_ids = list(imgs_with_anns.keys())
    random.shuffle(img_ids)
    img_ids = img_ids[:100]

    for img in data["images"]:
        if img["id"] in img_ids:
            new_images.append(img)

    for ann in data["annotations"]:
        if ann["image_id"] in imgs_with_anns:
            new_annotations.append(ann)

    data["images"] = new_images
    data["annotations"] = new_annotations

    with open("reduced_coco.json", "w") as f:
        json.dump(data, f)
