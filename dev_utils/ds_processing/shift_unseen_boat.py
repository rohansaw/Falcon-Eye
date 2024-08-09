# load coco json
import json
import os
import shutil

coco_json_path_train = "/scratch1/rsawahn/data/sw/coco/annotations/instances_train.json"
coco_json_path_val = "/scratch1/rsawahn/data/sw/coco/annotations/instances_val.json"
coco_json_path_test = "/scratch1/rsawahn/data/sw/coco/annotations/instances_test.json"
coco_json_path_train_new = os.path.join(os.path.dirname(coco_json_path_train), "train_new.json")
coco_json_path_val_new = os.path.join(os.path.dirname(coco_json_path_val), "val_new.json")
coco_json_path_test_new = os.path.join(os.path.dirname(coco_json_path_test), "test_new.json")

train_root = "/scratch1/rsawahn/data/sw/coco/train"
val_root = "/scratch1/rsawahn/data/sw/coco/val"
test_root = "/scratch1/rsawahn/data/sw/coco/test"

unseen_folder_name = "perlman"

coco_json = None

with open(coco_json_path_train, "r") as f:
    coco_json = json.load(f)
    
img_ids_to_shift = []
ann_ids_to_shift = []
fg_ids = set(ann["image_id"] for ann in coco_json["annotations"])
for img in coco_json["images"]:
    if unseen_folder_name in img["file_name"]:
        if img["id"] in fg_ids:
            img_ids_to_shift.append(img["id"])

for ann in coco_json["annotations"]:
    if ann["image_id"] in img_ids_to_shift:
        ann_ids_to_shift.append(ann["id"])

print(f"Moving {len(img_ids_to_shift)} images and {len(ann_ids_to_shift)} annotations to {unseen_folder_name} folder")
to_val_img_ids = img_ids_to_shift[:len(img_ids_to_shift)//2]
to_test_img_ids = img_ids_to_shift[len(img_ids_to_shift)//2:]
to_val_ann_ids = [ann["id"] for ann in coco_json["annotations"] if ann["image_id"] in to_val_img_ids]
to_test_ann_ids = [ann["id"] for ann in coco_json["annotations"] if ann["image_id"] in to_test_img_ids]

for img_id in to_val_img_ids:
    original_path = [img for img in coco_json["images"] if img["id"] == img_id][0]["file_name"]
    dest_path =  os.path.join(val_root, original_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.move(os.path.join(train_root, original_path), dest_path)
    
for img_id in to_test_img_ids:
    original_path = [img for img in coco_json["images"] if img["id"] == img_id][0]["file_name"]
    dest_path =  os.path.join(test_root, original_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.move(os.path.join(train_root, original_path), dest_path)

with open(coco_json_path_val, "r") as f:
    coco_json_val = json.load(f)
    coco_json_val["images"] = coco_json_val["images"] + [img for img in coco_json["images"] if img["id"] in to_val_img_ids]
    coco_json_val["annotations"] = coco_json_val["annotations"] + [ann for ann in coco_json["annotations"] if ann["id"] in to_val_ann_ids]
    
with open(coco_json_path_val_new, "w") as f:
    json.dump(coco_json_val, f)
    
with open(coco_json_path_test, "r") as f:
    coco_json_test = json.load(f)
    coco_json_test["images"] = coco_json_test["images"] + [img for img in coco_json["images"] if img["id"] in to_test_img_ids]
    coco_json_test["annotations"] = coco_json_test["annotations"] + [ann for ann in coco_json["annotations"] if ann["id"] in to_test_ann_ids]
    
with open(coco_json_path_test_new, "w") as f:
    json.dump(coco_json_test, f)
    
coco_json["images"] = [img for img in coco_json["images"] if img["id"] not in img_ids_to_shift]
coco_json["annotations"] = [ann for ann in coco_json["annotations"] if ann["id"] not in ann_ids_to_shift]
with open(coco_json_path_train_new, "w") as f:
    json.dump(coco_json, f)