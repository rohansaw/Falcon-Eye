import json
import random
import os

amounts = [0.5]
json_path = "/scratch1/rsawahn/data/sds_sliced_new/coco/annotations/instances_val.json"

for amount_neg in amounts:
    json_out_path = os.path.join(os.path.dirname(json_path), os.path.splitext(os.path.basename(json_path))[0] + f"-{int(amount_neg * 100)}percent_bg.json")
    coco = None
    #load coco json from file
    with open(json_path, "r") as f:
        coco = json.load(f)

    print("laoded")
    bg_ids = []
    bg_ids_dict = {}
    
    with_anns = {}
    
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        with_anns[img_id] = True
        
    bg_ids = [img["id"] for img in coco["images"] if not img["id"] in with_anns]
    bg_ids_dict = {k: True for k in bg_ids}

    print(len(coco["images"]))
    print(len(bg_ids))
    print("counting done")
    #reduce the amount of background images
    num_fg_ids = len(coco["images"]) - len(bg_ids)
    print(num_fg_ids)
    print(int(num_fg_ids * amount_neg))
    num_bg_ids_to_sample = int(num_fg_ids * amount_neg)
    selected_bg_ids = random.sample(bg_ids, num_bg_ids_to_sample)
    selected_bg_ids_dict = {}
    for id in selected_bg_ids:
        selected_bg_ids_dict[id] = True
    
    print("creating")
    coco["images"] = [img for img in coco["images"] if img["id"] not in bg_ids_dict or img["id"] in selected_bg_ids_dict]
    coco["annotations"] = [ann for ann in coco["annotations"] if ann["image_id"] not in bg_ids_dict or ann["image_id"] in selected_bg_ids_dict]

    print(len(coco["images"]))
    with open(json_out_path, "w") as f:
        json.dump(coco, f)