from collections import OrderedDict
import itertools
from COCOevalMP import COCOevalMP
from pycocotools.coco import COCO
import numpy as np
from tidecv import TIDE, datasets
#from terminaltables import AsciiTable
import json
import os
from pathlib import Path
import cv2


coco_gt_json_path = "/scratch1/rsawahn/data/sw/coco/annotations/instances_val.json"
#coco_pred_json_path = "/scratch1/rsawahn/data/results_coco/ssd_lite_sw_sliced.bbox.json"
coco_pred_json_path = "/scratch1/rsawahn/data/results_coco/slicedTrain_fullEval/efficientdet_sw-sliced-new_new_fullImgEval/sahi_sf_overlap_full.json"
#coco_pred_json_path = "/home/rsawahn/thesis/tiny-od-on-edge/baselines/efficientdet/testdev_output_sds_val/detections_test-dev2017_test_results.json"
sds_classes = ('swimmer', 'boat', 'jetski', 'life_saving_appliances', 'buoy')
sw_classes=('boat',)
tide_plot_out_path = os.path.join(os.path.dirname(coco_pred_json_path), "tide_plot.png")

# !!! change to sds_classes for sds

# print num predictions in coco_pred_json_path
with open(coco_pred_json_path, "r") as f:
    preds = json.load(f)
    print(len(preds))

# if coco_pred_json_path contains ssd, then remove all predictions with less than 0.2 score
# if "ssd" in coco_pred_json_path:
#     preds = [pred for pred in preds if pred["score"] > 0.2] ##otherwise OOM
#     print(len(preds))


# save as fixed json
# coco_pred_json_path = os.path.join("fixed_coco_preds.json")
# with open(coco_pred_json_path, "w") as f:
#     json.dump(preds, f)
    
#print([pred["image_id"] for pred in preds])
    
# print num gt anns
with open(coco_gt_json_path, "r") as f:
    gt = json.load(f)
    print(len(gt["annotations"]))

# count = 0
# for img_entry in gt["images"]:
#     if count > 300:
#         break
#     anns = [ann for ann in gt["annotations"] if ann["image_id"] == img_entry["id"]]
#     # draw anns on image and save
#     img = cv2.imread(os.path.join("/scratch1/rsawahn/data/sw_sliced/coco/val/", img_entry["file_name"]))
#     for ann in anns:
#         x, y, w, h = [int(v) for v in ann["bbox"]]
#         img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     if len(anns) != 0:
#         count += 1
#         cv2.imwrite(f"test_images/{img_entry['id']}.jpg", img)
# exit()

# print(set([pred["category_id"] for pred in preds]))
# print(set([ann["category_id"] for ann in gt["annotations"]]))
# new_preds = []
# for pred in preds:
#     for p in pred:
#         if not p == []:
#             new_preds.append(p)
        
# preds = new_preds

for pred in preds:
    pred["category_id"] = pred["category_id"] - 1
    if pred["bbox"] == []:
        preds.remove(pred)
        
coco_pred_json_path = os.path.join("fixed_coco_preds.json")
with open(coco_pred_json_path, "w") as f:
    json.dump(preds, f)

# requires_fix = ["faster_rcnn", "sss", "detectors"]
# if "sds" in coco_pred_json_path and any([rf in coco_pred_json_path for rf in requires_fix]):
#     for pred in preds:
#         pred["category_id"] = pred["category_id"] + 1
        
#     coco_pred_json_path = os.path.join("fixed_coco_preds.json")
#     with open(coco_pred_json_path, "w") as f:
#         json.dump(preds, f)

# For Yolo
# pred_anns_id = [ann["image_id"] for ann in preds]
# gt_images_id = [img["id"] for img in gt["images"]]

# real_ids = set(pred_anns_id).intersection(set(gt_images_id))
# print(len(real_ids))

# filename_id = {Path(img["file_name"]).stem: img["id"] for img in gt["images"]}

# print(len(real_ids) != len(set(pred_anns_id)))

# if len(real_ids) != len(set(pred_anns_id)):
#     for pred in preds:
#         filename = pred["image_id"]
#         pred["image_id"] = filename_id[filename]
        
# coco_pred_json_path = os.path.join("fixed_coco_preds.json")
# with open(coco_pred_json_path, "w") as f:
#    json.dump(preds, f)

# if "sw" in coco_pred_json_path and "efficientdet" in coco_pred_json_path:
#     for pred in preds:
#         pred["category_id"] = pred["category_id"] -1
        
#     coco_pred_json_path = os.path.join("fixed_coco_preds.json")
#     with open(coco_pred_json_path, "w") as f:
#         json.dump(preds, f)

print(set([pred["category_id"] for pred in preds]))
print(set([ann["category_id"] for ann in gt["annotations"]]))

coco_api = COCO(coco_gt_json_path)
cat_ids = coco_api.getCatIds(catNms=sw_classes)
img_ids = coco_api.getImgIds()
coco_preds = coco_api.loadRes(coco_pred_json_path)
coco_eval = COCOevalMP(coco_api, coco_preds, "bbox")
coco_eval.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 20 ** 2], [20 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
coco_eval.params.areaRngLbl = ['all', 'tiny', 'small', 'medium', 'large']

coco_eval.params.catIds = cat_ids
coco_eval.params.imgIds = img_ids
coco_eval.params.maxDets = list((1, 10, 100, 300, 1000))
coco_eval.params.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
coco_eval.params.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)

coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

eval_results = OrderedDict()
# from https://github.com/facebookresearch/detectron2/
precisions = coco_eval.eval['precision']
# precision: (iou, recall, cls, area range, max dets)
assert len(cat_ids) == precisions.shape[2]

results_per_category = []
for idx, cat_id in enumerate(cat_ids):
    t = []
    # area range index 0: all area ranges
    # max dets index -1: typically 100 per image
    nm = coco_api.loadCats(cat_id)[0]
    precision = precisions[:, :, idx, 0, -1]
    precision = precision[precision > -1]
    if precision.size:
        ap = np.mean(precision)
    else:
        ap = float('nan')
    t.append(f'{nm["name"]}')
    t.append(f'{round(ap, 3)}')
    eval_results[f'{nm["name"]}_precision'] = round(ap, 3)

    # indexes of IoU  @50 and @75
    for iou in [0, 5]:
        precision = precisions[iou, :, idx, 0, -1]
        precision = precision[precision > -1]
        if precision.size:
            ap = np.mean(precision)
        else:
            ap = float('nan')
        t.append(f'{round(ap, 3)}')

    # indexes of area of small, median and large
    for area in [1, 2, 3, 4]:
        precision = precisions[:, :, idx, area, -1]
        precision = precision[precision > -1]
        if precision.size:
            ap = np.mean(precision)
        else:
            ap = float('nan')
        t.append(f'{round(ap, 3)}')
    results_per_category.append(tuple(t))

num_columns = len(results_per_category[0])
results_flatten = list(
itertools.chain(*results_per_category))
headers = [
'category', 'mAP', 'mAP_50', 'mAP_75', 'mAP_t', 'mAP_s',
'mAP_m', 'mAP_l'
]
# results_2d = itertools.zip_longest(*[results_flatten[i::num_columns]for i in range(num_columns)])
# table_data = [headers]
# table_data += [result for result in results_2d]
# table = AsciiTable(table_data)
# print(table.table)

gt_tide = datasets.COCO(coco_gt_json_path)
pred_tide = datasets.COCOResult(coco_pred_json_path)
tide = TIDE()
tide.evaluate(gt_tide, pred_tide, pos_threshold=0.5, mode=TIDE.BOX, name="YOLOv8n-SW")
tide.summarize() 
tide.plot("tide_plot.png")