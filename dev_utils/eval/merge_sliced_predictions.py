from collections import defaultdict
import os
import json
from pathlib import Path
from sahi.postprocess.combine import NMSPostprocess
import torch
from nms import nms, to_torch, torch_to_coco_pred, batched_nms
import cv2

img_path = "/scratch1/rsawahn/data/sw/coco/val/"
tiles_json = "/scratch1/rsawahn/data/sw_sliced/coco/annotations/instances_val.json"
original_json = "/scratch1/rsawahn/data/sw/coco/annotations/instances_val.json"
tiles_res_json = "/scratch1/rsawahn/data/results_coco/ssd_lite_sw_sliced.bbox.json"
out_folder = "/scratch1/rsawahn/data/results_coco/"

is_yolo = False
if "yolo" in tiles_res_json:
    is_yolo = True


def get_original_file_name_and_coords(file_name):
    filename_base = Path(file_name).stem
    return "".join(file_name.split("_")[:-4]), filename_base.split("_")[-4:]

def shift_res_ann(coords, res):
    x, y, x2, y2 = [int(v) for v in coords]
    shifted_res = res.copy()
    shifted_res["bbox"] = [
        res["bbox"][0] + x,
        res["bbox"][1] + y,
        res["bbox"][2],
        res["bbox"][3]
    ]
    return shifted_res

with open(tiles_json, "r") as f:
    tiles_coco = json.load(f)

with open(tiles_res_json, "r") as f:
    tiles_res_coco = json.load(f)
    
with open(original_json, "r") as f:
    original_coco = json.load(f)

tile_coords = {}
tile_img = {}

img_fn_to_full_path = {}
# go trough all files including subdirs in img_path
for root, dirs, files in os.walk(img_path):
    for file in files:
        img_fn_to_full_path[Path(file).stem] = os.path.join(root, file)

yolo_lookup_tile_id = {}

for tile in tiles_coco["images"]:
    file_name, coords = get_original_file_name_and_coords(tile["file_name"])
    
    if is_yolo:
        yolo_lookup_tile_id[Path(tile["file_name"]).stem] = tile["id"]
    
    tile_img[tile["id"]] = file_name
    tile_coords[tile["id"]] = coords

img_tile_res = {}
filename_id = {Path(img["file_name"]).stem: img["id"] for img in original_coco["images"]}

for res in tiles_res_coco:
    tile_id = res["image_id"]
    
    if is_yolo:
        tile_id = yolo_lookup_tile_id[tile_id]
    
    img_id = tile_img[tile_id]
    
    current_img_tile_resses = img_tile_res.get(img_id, {})
    current_img_tile_resses[tile_id] = res
    img_tile_res[img_id] = current_img_tile_resses

    
predictions = []
for img_id, img_tiles in img_tile_res.items():
    shifted_preids_img = []
    for tile_res in img_tiles.values():
        tile_id = tile_res["image_id"]
        
        if is_yolo:
            tile_id = yolo_lookup_tile_id[tile_id]
        tile_position = tile_coords[tile_id]
        shifted_res = shift_res_ann(tile_position, tile_res)
        shifted_res["image_id"] = int(filename_id[img_id])
        shifted_preids_img.append(shifted_res)
        predictions.append(shifted_res)
        
    # load the image and visualize the predictions
    file_name = [img["file_name"] for img in original_coco["images"] if img["id"] == int(filename_id[img_id])][0]
    path = os.path.join(img_path, file_name)
    
    # visualize the predictions
    img = cv2.imread(path)
    for pred in shifted_preids_img:
        x, y, w, h = [int(v) for v in pred["bbox"]]
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # save the image
    cv2.imwrite(f"test_images/{img_id}.jpg", img)
    

print(len(set([img["image_id"] for img in predictions])))

# save preditions in json
# out path is the same as the input path, but with _merged.json
tile_file_name = Path(tiles_res_json).stem
out_path = os.path.join("merged_dets", f"merged{tile_file_name}.json")
with open(out_path, "w") as f:
    json.dump(predictions, f)

preds_by_img = defaultdict(list)

for pred in predictions:
    preds_by_img[pred["image_id"]].append(pred)

processed_preds = []

# for img_id in preds_by_img:
#     preds_torch = to_torch(preds_by_img[img_id])
#     #keep = nms(preds_torch, match_metric="IOU", match_threshold=0.5)
#     keep = preds_torch
#     filtered_preds = preds_torch[keep]
#     coco_keep = torch_to_coco_pred(filtered_preds, img_id)
#     processed_preds.extend(coco_keep)
    
# with open(os.path.join(out_folder, "merged_nms.json"), "w") as f:
#     json.dump(processed_preds, f)