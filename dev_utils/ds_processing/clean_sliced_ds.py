# remove all images that are not in the sliced dataset
import json
import os

image_dir = "/scratch1/rsawahn/data/sds_sliced_centered/coco/val"
labels_path = "/scratch1/rsawahn/data/sds_sliced_centered/coco/annotations/instances_val.json"

with open(labels_path, "r") as f:
    coco = json.load(f)
    
ann_count = {}
for ann in coco["annotations"]:
    if ann["image_id"] in ann_count:
        ann_count[ann["image_id"]] += 1
    else:
        ann_count[ann["image_id"]] = 1
        
k_path = {}
for img in coco["images"]:
    # replace ending ".jpg" to ".jpeg"
    k_path[img["id"]] = img["file_name"].replace(".jpg", ".jpeg")

to_delete = []
num_without_ann = 0
for img in coco["images"]:
    if img["id"] not in ann_count:
        to_delete.append(img["id"])
        num_without_ann += 1


images_kept = len(coco["images"]) - len(to_delete)
amount_keep_bg_only = 0.5 * images_kept
end_index = len(to_delete) - int(amount_keep_bg_only)

# shuffle to_delete
import random
random.shuffle(to_delete)
to_delete = to_delete[:end_index]
        
print("to_delete: ", len(to_delete))
print("should be: ", len(coco["images"]) - len(to_delete))

print("images without annotations: ", num_without_ann)
print(len(coco["images"])-num_without_ann)
count = 0
print("total in folder: ", len(os.listdir(image_dir)))
print("total in dataset: ", len(coco['images']))

for k in to_delete:
    p = os.path.join(image_dir, k_path[k])
    print(p)
    try:
        os.remove(p)
        count += 1
    except FileNotFoundError:
        print("File not found")
        pass
    except Exception as e:
        print(e)
        print("error deleting: ", k)
        break

# save the new json
def edit_end(img):
    img["file_name"] = img["file_name"].replace(".jpeg", ".jpg")
    return img

coco["images"] = [edit_end(img) for img in coco["images"] if img["id"] not in to_delete]

new_labels_path = labels_path.replace(".json", "_cleaned.json")
with open(new_labels_path, "w") as f:
    json.dump(coco, f)

print("total in folder: ", len(os.listdir(image_dir)))
