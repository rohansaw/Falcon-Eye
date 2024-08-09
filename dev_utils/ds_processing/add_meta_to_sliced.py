import json
import os

# load coco json file
json_file_sliced = "/data/sw_sliced_centered/coco/annotations/instances_train-0percent_bg.json"
json_file_original = "/data/sw/coco/annotations/instances_train.json"

meta_keys = ["altitude", "focal_length"]

with open(json_file_sliced) as f:
    coco_sliced = json.load(f)
    
with open(json_file_original) as f:
    coco_original = json.load(f)
    
# create file_name dict indexed by original file_name
file_name_dict = {}
for img in coco_original["images"]:
    file_path = img["file_name"]
    file_name = os.path.basename(file_path)
    file_name, ext = os.path.splitext(file_name)
    file_name_dict[file_name] = img
    
for img in coco_sliced["images"]:
    file_name = img["file_name"]
    original_fn = file_name.split("_")[:-4]
    original_fn = "_".join(original_fn)
    
    original_meta = file_name_dict[original_fn]
    for k in meta_keys:
        img[k] = original_meta.get(k)
    
# save the new json file
output_file = os.path.join(os.path.dirname(json_file_sliced), os.path.basename(json_file_sliced).replace(".json", "_meta.json"))
with open(output_file, "w") as f:
    json.dump(coco_sliced, f)