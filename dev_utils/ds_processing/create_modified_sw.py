import json
train_coco = None
val_coco = None
test_coco = None
train_json = "/scratch1/rsawahn/data/sw_sliced/coco/annotations/instances_train-50percent_bg.json"
val_json = "/scratch1/rsawahn/data/sw_sliced/coco/annotations/instances_val-50percent_bg.json"
test_json = "/scratch1/rsawahn/data/sw_sliced/coco/annotations/instances_test-50percent_bg.json"
with open(train_json, "r") as f:
    train_coco = json.load(f)

with open(val_json, "r") as f:
    val_coco = json.load(f)
    
with open(test_json, "r") as f:
    test_coco = json.load(f)
    
all_coco = [train_coco, val_coco, test_coco]
ds_names = ["train-50percent_bg", "val-50percent_bg", "test-50percent_bg"]

print(train_coco.keys(), val_coco.keys(), test_coco.keys())
print(train_coco["categories"], val_coco["categories"], test_coco["categories"])

# print all keys used in annotations as a set
keys_to_delete = ['altitude', 'focal_length', 'beaufort', 'camera_model']
for ds in all_coco:
    for img in ds["images"]:
        img_keys_to_delete = []
        for key in img.keys():
            if key in keys_to_delete:
                img_keys_to_delete.append(key)
        for key in img_keys_to_delete:
            del img[key]
            
    for ann in ds["annotations"]:
        if ann["category_id"] == 0:
            ann["category_id"] = 1
    
    # add "ignore" category with id 0
    ds["categories"] = [{"id": 0, "name": "ignore", "supercategory": "ignore"}, {"id": 1, "name": "boat", "supercategory": "boat"}]
    
# save mdofied jsons with suffix exp_cleaned
for ds, ds_name in zip(all_coco, ds_names):
    with open(f"/scratch1/rsawahn/data/sw_sliced/coco/annotations/instances_{ds_name}_exp_cleaned.json", "w") as f:
        json.dump(ds, f)