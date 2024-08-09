import json
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import Metrics
from PIL import Image, ImageDraw
from utils.visualization import draw_prediction
import time
from torchvision.transforms import v2


class COCOEvaluator:
    def __init__(self, results_path, gt_path, classes):
        self.results_path = results_path
        self.gt_path = gt_path
        self.classes = classes
        self.num_classes = len(classes)
        self.gt = self.load_json(gt_path)
        self.results = self.load_json(results_path)
        self.gt_by_image_id = self.group_annotations_by_image(self.gt["annotations"])
        self.results_by_image_id = self.group_annotations_by_image(self.results, verbose=True)
        self.image_metadata = {
            img["id"]: (img["height"], img["width"]) for img in self.gt["images"]
        }
        
        self.file_names = {img["id"]: img["file_name"] for img in self.gt["images"]}

    def load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def group_annotations_by_image(self, annotations, verbose=False):
        grouped = {}
        for annotation in annotations:
            grouped.setdefault(annotation["image_id"], []).append(annotation)
        return grouped

    def evaluate(self, scale_factor, confidence_thresholds, tolerance=0):
        ap_per_class = {}
        all_true_positives = 0
        all_false_positives = 0
        all_false_negatives = 0
        
        all_by_size_collected = {
            "tiny": {"TP": 0, "FN": 0},
            "small": {"TP": 0, "FN": 0},
            "medium": {"TP": 0, "FN": 0},
            "large": {"TP": 0, "FN": 0},
        }

        for _, class_id in self.classes:
            all_precisions, all_recalls, all_f1, class_tp, class_fp, class_fn, all_by_size = self.evaluate_class(
                class_id, scale_factor, confidence_thresholds, tolerance
            )
            ap_per_class[class_id] = self.calculate_ap(all_precisions, all_recalls)
            all_true_positives += class_tp
            all_false_positives += class_fp
            all_false_negatives += class_fn
            
            for item in all_by_size:
                for size, entry in item.items():
                    for m_type, val in entry.items():
                        all_by_size_collected[size][m_type] += val

        if all_true_positives + all_false_positives == 0:
            micro_precision = 0
        else:
            micro_precision = all_true_positives / (all_true_positives + all_false_positives)
        
        if all_true_positives + all_false_negatives == 0:
            micro_recall = 0
        else:
            micro_recall = all_true_positives / (all_true_positives + all_false_negatives)
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

        print(f"Micro-averaged Precision: {micro_precision}")
        print(f"Micro-averaged Recall: {micro_recall}")
        print(f"Micro-averaged F1: {micro_f1}")
        
        print(all_by_size_collected)
        
        recalls_by_size = {}
        # calculate recall for each size
        for size, entry in all_by_size_collected.items():
            recall = entry["TP"] / (entry["TP"] + entry["FN"]) if (entry["TP"] + entry["FN"]) > 0 else 0
            print(f"Recall for size {size}: {recall}")
            recalls_by_size[size] = recall

        return np.mean(list(ap_per_class.values())), micro_precision, micro_recall, micro_f1, recalls_by_size

    def evaluate_class(self, class_id, scale_factor, confidence_thresholds, tolerance):
        all_precisions, all_recalls, all_f1, all_by_size = [], [], [], []
        class_true_positives = 0
        class_false_positives = 0
        class_false_negatives = 0

        for confidence_threshold in confidence_thresholds:
            metrics = Metrics(
                scale_factor=scale_factor,
                classes=self.classes,
                treshold=0.5,
                tolerance=tolerance,
            )
            
            for image_id, gt_annotations in self.gt_by_image_id.items():
                gt_filtered = [
                    ann for ann in gt_annotations if ann["category_id"] == class_id
                ]
                predictions = self.results_by_image_id.get(image_id, [])
                print(predictions)
                predictions_filtered = [
                    pred
                    for pred in predictions
                    if pred["category_id"] == class_id
                    and pred["score"] >= confidence_threshold
                ]
                img_size = self.image_metadata[image_id]
                
                prediction_grid = self.convert_to_grid(
                    img_size, scale_factor, 1, predictions_filtered
                )
                
                y_true = self.convert_to_cocowrapper(gt_filtered)
                
                # t = time.time()
                # img_path = os.path.join("/data/sw/coco/val/", self.file_names[image_id])
                # img = Image.open(img_path)
                # save_path = f"/home/thesis/tiny-od-on-edge/detector/pred_images/all/baseline_res_{t}.png"
                # if not os.path.exists(f"/home/thesis/tiny-od-on-edge/detector/pred_images/all"):
                #     os.makedirs(f"/home/thesis/tiny-od-on-edge/detector/pred_images/all")
                    
                # grid_path = f"/home/thesis/tiny-od-on-edge/detector/pred_images/all/grid_{t}.txt"
                # np.savetxt(grid_path, prediction_grid[1].numpy(), fmt="%d")
            
                # image to torch
                # img = v2.ToTensor()(img)
                # std = [1, 1, 1]
                # mean = [0, 0, 0]
                # 
                # #y_merged = self.convert_to_cocowrapper(gt_filtered + predictions_filtered)
                # draw_prediction(prediction_grid, img, save_path, 0.1, mean, std, False, [y_true])
                
                
                metrics.process_batch(prediction_grid.unsqueeze(0), [y_true])

            metrics_results = metrics.compute_metrics()
            all_precisions.append(metrics_results["precision"])
            all_recalls.append(metrics_results["recall"])
            all_by_size.append(metrics.compute_sized_metrics())

            class_true_positives += metrics.tp_fp_tn_fn_counts["TP"]
            class_false_positives += metrics.tp_fp_tn_fn_counts["FP"]
            class_false_negatives += metrics.tp_fp_tn_fn_counts["FN"]

            if metrics_results["precision"] + metrics_results["recall"] == 0:
                f1 = 0
            else:
                f1 = 2 * (metrics_results["precision"] * metrics_results["recall"]) / (metrics_results["precision"] + metrics_results["recall"])
            all_f1.append(f1)
            print(
                f"Class: {class_id}, Confidence Threshold: {confidence_threshold}, Precision: {metrics_results['precision']}, Recall: {metrics_results['recall']}, F1: {f1}"
            )

        return all_precisions, all_recalls, all_f1, class_true_positives, class_false_positives, class_false_negatives, all_by_size

    @staticmethod
    def convert_to_grid(img_sz, scale_factor, num_classes, predictions):
        grid_h = int(img_sz[0] * scale_factor)
        grid_w = int(img_sz[1] * scale_factor)
        grid_shape = (num_classes + 1, grid_h, grid_w)
        grid = torch.zeros(grid_shape)
        # Set all feature map cells to background initially
        grid[0, :, :] = 1

        for pred in predictions:
            class_id = pred["category_id"] + 1
            bbox = pred["bbox"]
            x, y, w, h = bbox

            x_scaled = max(0, int(np.floor(x * scale_factor)))
            y_scaled = max(0, int(np.floor(y * scale_factor)))
            x2_scaled = min(grid_w -1, int(np.floor((x + w) * scale_factor)))
            y2_scaled = min(grid_h -1, int(np.floor((y + h) * scale_factor)))

            center_x = (x_scaled + x2_scaled) // 2
            center_y = (y_scaled + y2_scaled) // 2
            
            if center_x >= grid_w or center_y >= grid_h:
                print("skip")
                continue

            grid[1, center_y, center_x] = 1
            # grid[class_id, y_scaled:y2_scaled, x_scaled:x2_scaled] = 1  # Flipping x and y here??
            grid[0, y_scaled:y2_scaled, x_scaled:x2_scaled] = 0
            #grid[0, center_y, center_x] = 0
        return grid

    @staticmethod
    def convert_to_cocowrapper(annotations):
        y_true = {"boxes": [], "labels": []}
        for ann in annotations:
            bbox = ann["bbox"]
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            y_true["boxes"].append(torch.tensor([x1, y1, x2, y2]))
            y_true["labels"].append(torch.tensor(0))
        return y_true

    @staticmethod
    def calculate_ap(precisions, recalls):
        sorted_indices = np.argsort(recalls)
        sorted_recalls = np.array(recalls)[sorted_indices]
        sorted_precisions = np.array(precisions)[sorted_indices]
        return np.trapz(sorted_precisions, sorted_recalls)


# Usage
#results_path =  "/data/results_coco/coco_ssd_sds.bbox.json"

# if "sds" in results_path and "efficientdet" in results_path:
#     with open(results_path, "r") as f:
#         preds = json.load(f)
#     for pred in preds:
#         pred["category_id"] = pred["category_id"] +1
        
#     results_path = os.path.join("fixed_coco_preds.json")
#     with open(results_path, "w") as f:
#         json.dump(preds, f)


sw_classes = [("swimmer", 0)]
sds_classes = [("swimmer", 1), ('boat', 2), ('jetski', 3), ('life_saving_appliances', 4), ('buoy',5)]
scale_factor = 0.125
#confidence_thresholds = np.linspace(0, 1, num=11)
#confidence_thresholds = [0.1]
tolerance = 4

def evaluate(results_path, gt_path, classes, scale_factor, confidence_thresholds, tolerance):

    # activate for yolo results, since they are using the wrong ids
    if "yolo" in results_path:
        with open(results_path, "r") as f:
            preds = json.load(f)
        with open(gt_path, "r") as f:
            gt = json.load(f)
            
        pred_anns_id = [ann["image_id"] for ann in preds]
        gt_images_id = [img["id"] for img in gt["images"]]

        real_ids = set(pred_anns_id).intersection(set(gt_images_id))

        filename_id = {Path(img["file_name"]).stem: img["id"] for img in gt["images"]}

        if len(real_ids) != len(set(pred_anns_id)):
            for pred in preds:
                filename = pred["image_id"]
                pred["image_id"] = filename_id[filename]
                
        results_path = os.path.join("fixed_coco_preds.json")
        # write to results
        with open(results_path, "w") as f:
            json.dump(preds, f)

        # check if the classes are correct
        with open(results_path, "r") as f:
            preds = json.load(f)
            print(set([pred["category_id"] for pred in preds]))

        print(set([img["category_id"] for img in gt["annotations"]]))

        pred_anns_id = [ann["image_id"] for ann in preds]
        gt_images_id = [img["id"] for img in gt["images"]]
        real_ids = set(pred_anns_id).intersection(set(gt_images_id))
        print(len(real_ids))
        print(len(set(pred_anns_id)))
        print(len(set(gt_images_id)))

    if "sw" in results_path and "efficientdet" in results_path:
        with open(results_path, "r") as f:
            preds = json.load(f)
        for pred in preds:
            pred["category_id"] = pred["category_id"] -1
            
        results_path = os.path.join("fixed_coco_preds.json")
        with open(results_path, "w") as f:
            json.dump(preds, f)
            
    if "sds" in results_path and any([x in results_path for x in ["ssd", "faster", "detectors"]]):
        with open(results_path, "r") as f:
            preds = json.load(f)
            
        # print set of category ids
        print(set([pred["category_id"] for pred in preds]))
        
        for pred in preds:
            pred["category_id"] = pred["category_id"] +1
            
        results_path = os.path.join("fixed_coco_preds.json")
        with open(results_path, "w") as f:
            json.dump(preds, f)

    evaluator = COCOEvaluator(results_path, gt_path, classes)
    map_value, micro_precision, micro_recall, micro_f1, recalls_by_size = evaluator.evaluate(scale_factor, confidence_thresholds, tolerance)
    print(f"Mean Average Precision (mAP): {map_value}")
    print(f"Micro-averaged Precision: {micro_precision}")
    print(f"Micro-averaged Recall: {micro_recall}")
    print(f"Micro-averaged F1: {micro_f1}")
    return {
        "mAP": map_value,
        "precision": micro_precision,
        "recall": micro_recall,
        "f1": micro_f1,
        "recalls_by_size": recalls_by_size,
    }

sds_gt = "/data/sds/coco/annotations/instances_val.json"
sw_gt = "/data/sw/coco/annotations/instances_val.json"

configs = [
    {
        "results_path": "/data/results_coco/coco_sw._faster_rcnnbbox.json",
        "gt_path": sw_gt,
        "type": "sw",
        "name": "FasterRCNN",
    },
    # {
    #     "results_path": "/data/results_coco/slicedTrain_fullEval/efficientdet_sw-sliced-new_new_fullImgEval/sahi_sf_overlap_full.json",
    #     "gt_path": sw_gt,
    #     "type": "sw",
    #     "name": "EfficientDet",
    # },
    # {
    #     "results_path": "/data/results_coco/slicedTrain_fullEval/efficientdet_sds-sliced-new_new_fullImgEval/sahi_sf_overlap_full.json",
    #     "gt_path": sds_gt,
    #     "type": "sds",
    #     "name": "EfficientDet",
    # },
    # {
    #     "results_path": "/data/results_coco/slicedTrain_fullEval/ssd_sw-sliced-new_new_fullImgEval/sahi_sf_overlap_full.json",
    #     "gt_path": sw_gt,
    #     "type": "sw",
    #     "name": "SSD",
    # },
    # {
    #     "results_path": "/data/results_coco/slicedTrain_fullEval/ssd_sds-sliced-new_new_fullImgEval/sahi_sf_overlap_full.json",
    #     "gt_path": sds_gt,
    #     "type": "sds",
    #     "name": "SSD",
    # },
    # {
    #     "results_path": "/data/results_coco/slicedTrain_fullEval/detectors_sw-sliced-new_fullImgEval/sahi_sf_overlap_full.json",
    #     "gt_path": sw_gt,
    #     "type": "sw",
    #     "name": "DetectoRS",
    # },
    # {
    #     "results_path": "/data/results_coco/slicedTrain_fullEval/detectors_sds-sliced-new_fullImgEval/sahi_sf_overlap_full.json",
    #     "gt_path": sds_gt,
    #     "type": "sds",
    #     "name": "DetectoRS",
    # },
    # {
    #     "results_path": "/data/results_coco/slicedTrain_fullEval/sw_sliced_new_yolo/sw_sliced_new_yolov8_coco.json",
    #     "gt_path": sw_gt,
    #     "type": "sw",
    #     "name": "YOLOv8n",
    # },
    # {
    #     "results_path": "/data/results_coco/slicedTrain_fullEval/sds_sliced_new_yolo/sds_sliced_new_yolov8_coco.json",
    #     "gt_path": sds_gt,
    #     "type": "sds",
    #     "name": "YOLOv8n",
    # },
    # {
    #     "results_path": "/data/results_coco/slicedTrain_fullEval/faster-rcnn_sw-sliced-new_new_fullImgEval/sahi_sf_overlap_full.json",
    #     "gt_path": sw_gt,
    #     "type": "sw",
    #     "name": "FasterRCNN",
    # },
    # {
    #     "results_path": "/data/results_coco/slicedTrain_fullEval/faster-rcnn_sds-sliced-new_new_fullImgEval/sahi_sf_overlap_full.json",
    #     "gt_path": sds_gt,
    #     "type": "sds",
    #     "name": "FasterRCNN",
    # },
]

# check all paths exist
for config in configs:
    print(config["results_path"])
    assert os.path.exists(config["results_path"])
    assert os.path.exists(config["gt_path"])
    
# create a precision recall plot for all detectors and both datasets, make it easily comparable/understandable

thresholds = np.linspace(0, 1, 100)

colors = {
    "EfficientDet": "r",
    "SSD": "g",
    "DetectoRS": "b",
    "YOLOv8n": "c",
    "FasterRCNN": "m",

}

results = {}
plt.figure(figsize=(8, 6))
for config in configs:
    results_path = config["results_path"]
    gt_path = config["gt_path"]
    name = config["name"]
    t = config["type"]
    
    print(f"Evaluating {name} on {t} dataset")
    precisions = []
    recalls = []
    
    recalls_by_size = {}

    for threshold in thresholds:
        res = evaluate(results_path, gt_path, sds_classes if t == "sds" else sw_classes, scale_factor, [threshold], tolerance)
        precisions.append(res["precision"])
        recalls.append(res["recall"])
        for size, recall in res["recalls_by_size"].items():
            if not size in recalls_by_size:
                recalls_by_size[size] = []
            recalls_by_size[size].append(recall)
        
    json_res = {
        "model_name": results_path,
        "precisions": precisions,
        "recalls": recalls,
        "recalls_by_size": recalls_by_size,
    }

    fn = f"pr_{name}_{t}.json"

    with open(f"predictions_store_baselines/{fn}_sliced.json", "w") as f:
        json.dump(json_res, f)
        
    results[f"{name}_{t}"] = {
        "precisions": precisions,
        "recalls": recalls,
    }
    
    linestyle = "-" if t == "sds" else "--"
    color = colors[name]
    #plt.plot(recalls, precisions, label=f"{name}_{t}", color=color, linestyle=linestyle)

# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend()
# plt.grid(True)
# plt.savefig("precision_recall_curve.png")
    
    