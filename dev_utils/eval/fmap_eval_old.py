import json
from PIL import Image
import numpy as np

def load_coco(path):
    with open(path, 'r') as f:
        return json.load(f)

def eval_image(anns_gt, anns_pred, treshold, downsampling_factor, num_classes, w, h):
    TP, FP, FN , TN = 0, 0, 0,0
    fm_preds = convert_anns(anns_pred, w,h, downsampling_factor, num_classes)
    fm_gt = convert_anns(anns_gt, w,h, downsampling_factor, num_classes)
    
    for ann in anns_gt:
        bbox = ann['bbox']
        bbox_cls = ann['category_id']
        fm_pred = fm_preds[bbox_cls]
        fm_gt = convert_bbox(bbox, w, h, downsampling_factor, 1)
        
        # check if any in fm_pred is above treshold where fm_gt is 1
        found = False
        for idx_r,row in enumerate(fm_pred):
            for idx_v,val in enumerate(row):
                if val >= treshold and fm_gt[idx_r][idx_v] == 1:
                    found = True
                    break
        if found:
            TP += 1
        else:
            FN += 1
            
    for cls_idx, map in enumerate(fm_preds):
        for idx_r, row in enumerate(map):
            for idx_v, val in enumerate(row):
                if val >= treshold and fm_gt[cls_idx][idx_r][idx_v] == 0:
                    FP += 1
                else:
                    TN += 1
    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN
    }
            
def convert_bbox(bbox, score, w, h, downsampling_factor, padding_size = 0):
    # return a featuremap representation of the downsampled image with bbox, invariant of a class
    fmap = np.zeros((int(h/downsampling_factor), int(w/downsampling_factor)))
    top_left = (max(0, int(bbox[0]/downsampling_factor)), max(0, int(bbox[1]/downsampling_factor)))
    bottom_right = (int((bbox[0]+bbox[2])/downsampling_factor), int((bbox[1]+bbox[3])/downsampling_factor))
    top_left = (max(0, top_left[0]-padding_size), max(0, top_left[1]-padding_size))
    bottom_right = (min(w/downsampling_factor, bottom_right[0]+padding_size), min(h/downsampling_factor, bottom_right[1]+padding_size))
    # ensure correct x,y
    fmap[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = score
    return fmap
    

def convert_anns(anns, w, h, downsampling_factor, num_classes):
    feat_maps = np.array([np.zeros((int(w/downsampling_factor), int(h/downsampling_factor))) for _ in range(num_classes)])
    for ann in anns:
        bbox = ann['bbox']
        cls = ann['category_id']
        score = ann['score'] if 'score' in ann else 1
        feat_maps[cls] += convert_bbox(bbox, score, w, h, downsampling_factor)
    return feat_maps

def make_metrics(results):
    metrics = {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "TN": 0
    }
    for result in results:
        metrics["TP"] += result["TP"]
        metrics["FP"] += result["FP"]
        metrics["FN"] += result["FN"]
        metrics["TN"] += result["TN"]
    
    metrics["precision"] = metrics["TP"] / (metrics["TP"] + metrics["FP"])
    metrics["recall"] = metrics["TP"] / (metrics["TP"] + metrics["FN"])
    metrics["f1"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])

    return metrics

def eval(gt_path, pred_path, treshold, downsampling_factor):
    gt_coco = load_coco(gt_path)
    pred_coco = load_coco(pred_path)
    w, h = 512, 512
    results = []
    for img in pred_coco['images']:
        anns_gt = [ann for ann in gt_coco['annotations'] if ann['image_id'] == img['id']]
        anns_pred = [ann for ann in pred_coco['annotations'] if ann['image_id'] == img['id']]
        results.append(eval_image(anns_gt, anns_pred, treshold, downsampling_factor, len(pred_coco['categories']), w, h))
        
    print(make_metrics(results))
        
# ToDo Resize to w,h the bbox anns