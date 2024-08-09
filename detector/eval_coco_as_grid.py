import json
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import Metrics


class COCOEvaluator:
    def __init__(self, results_path, gt_path, classes):
        self.results_path = results_path
        self.gt_path = gt_path
        self.classes = classes
        self.num_classes = len(classes)
        self.gt = self.load_json(gt_path)
        self.results = self.load_json(results_path)
        self.gt_by_image_id = self.group_annotations_by_image(self.gt["annotations"])
        self.results_by_image_id = self.group_annotations_by_image(self.results)
        self.image_metadata = {
            img["id"]: (img["height"], img["width"]) for img in self.gt["images"]
        }

    def load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def group_annotations_by_image(self, annotations):
        grouped = {}
        for annotation in annotations:
            grouped.setdefault(annotation["image_id"], []).append(annotation)
        return grouped

    def evaluate(self, scale_factor, confidence_thresholds, tolerance):
        ap_per_class = {}
        for _, class_id in self.classes:
            all_precisions, all_recalls, all_f1 = self.evaluate_class(
                class_id, scale_factor, confidence_thresholds, tolerance
            )
            ap_per_class[class_id] = self.calculate_ap(all_precisions, all_recalls)
        return np.mean(list(ap_per_class.values()))

    def evaluate_class(self, class_id, scale_factor, confidence_thresholds, tolerance):
        all_precisions, all_recalls, all_f1 = [], [], []
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
                metrics.process_batch(prediction_grid.unsqueeze(0), [y_true])

            metrics_results = metrics.compute_metrics()
            all_precisions.append(metrics_results["precision"])
            all_recalls.append(metrics_results["recall"])
           
            f1 = 2 * (metrics_results["precision"] * metrics_results["recall"]) / (metrics_results["precision"] + metrics_results["recall"])
            all_f1.append(f1)
            print(
                f"Class: {class_id}, Confidence Threshold: {confidence_threshold}, Precision: {metrics_results['precision']}, Recall: {metrics_results['recall']}, F1: {f1}"
            )

        return all_precisions, all_recalls, all_f1

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

            grid[1, center_y, center_x] = 1
            # grid[class_id, y_scaled:y2_scaled, x_scaled:x2_scaled] = 1  # Flipping x and y here??
            grid[0, y_scaled:y2_scaled, x_scaled:x2_scaled] = 0
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
results_path =  "/data/data/results_coco/yolo_sds1.json"
gt_path = "/data/data/sds/coco/annotations/instances_val.json"

# activate for yolo results, since they are using the wrong ids
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

classes = [("swimmer", 0)]
sds_classes = [("swimmer", 1), ('boat', 2), ('jetski', 3), ('life_saving_appliances', 4), ('buoy',5)]
scale_factor = 0.125
#confidence_thresholds = np.linspace(0, 1, num=11)
confidence_thresholds = [0.5]
tolerance = 2

evaluator = COCOEvaluator(results_path, gt_path, sds_classes)
map_value = evaluator.evaluate(scale_factor, confidence_thresholds, tolerance)
print(f"Mean Average Precision (mAP): {map_value}")
