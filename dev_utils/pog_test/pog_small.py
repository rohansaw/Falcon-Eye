import json
import os
import random
import shutil

random.seed(0)

json_p = "/scratch1/rsawahn/data/sw_sliced_centered/coco/annotations/instances_val.json"
num_samples = 500

with open(json_p, 'r') as f:
    coco = json.load(f)
    print(len(coco["images"]))
    
    # randomly sample 500 els from coco["images"]
    #sampled = random.sample(coco["images"], num_samples)
    #coco["images"] = sampled
    
    #ids = [img["id"] for img in sampled]
    #coco["annotations"] = [ann for ann in coco["annotations"] if ann["image_id"] in ids]
    #print(len(coco["images"]))
    
    count = 0
    goal = 1285
    
    # count annotations for each image in coco
    ann_count = {}
    for ann in coco["annotations"]:
        if ann["image_id"] in ann_count:
            ann_count[ann["image_id"]] += 1
        else:
            ann_count[ann["image_id"]] = 1
        
    
    for img in coco["images"]:
        # copy img from original path to new root with same path
        if img["id"] in ann_count and ann_count[img["id"]] != 0:
            continue
        if count > goal:
            break
        p = img["file_name"]
        p_old = os.path.join("/scratch1/rsawahn/data/sw_sliced_centered/coco/val", img["file_name"])
        p_new = os.path.join("/scratch1/rsawahn/misc/sw_sliced_centered_small/val2", img["file_name"])
        os.makedirs(os.path.dirname(p_new), exist_ok=True)
        shutil.copy(p_old, p_new)
        count += 1
    
    #with open("/scratch1/rsawahn/misc/pog_sliced_small/annotations/instances_val.json", 'w') as f:
    #    json.dump(coco, f)