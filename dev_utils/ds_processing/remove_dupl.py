import json

json_path = "/scratch1/rsawahn/sw_original/coco/annotations/instances_train.json"
json_out_path = "/scratch1/rsawahn/sw_original/coco/annotations/instances_train_no_dupl.json"

# load coco from json as json obj
with open(json_path, 'r') as f:
    coco = json.load(f)
    
# remove duplicates
seen = {}
to_remove = []

for img in coco["images"]:
    if not img["file_name"] in seen:
        seen[img["file_name"]] = img
    else:
        to_remove.append(img["id"])
        
coco["annotations"] = [ann for ann in coco["annotations"] if ann["image_id"] not in to_remove]
coco["images"] = [img for img in coco["images"] if img["id"] not in to_remove]

# save to new json
with open(json_out_path, 'w') as f:
    json.dump(coco, f)