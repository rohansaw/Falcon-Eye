import numpy as np
import torch
from enum import Enum

default_sizes = [("tiny", 20**2), ("small", 32**2), ("medium", 96**2), ("large", None)]


class CountType(Enum):
    TP = "TP"
    FP = "FP"
    TN = "TN"
    FN = "FN"


class Metrics:
    def __init__(self, scale_factor, tolerance, treshold, classes, sizes=default_sizes, use_softmax=True, binary=False):
        self.scale_factor = scale_factor
        self.tolerance = tolerance
        self.treshold = treshold
        self.classes = classes # tuple ("class_name", class_id)
        self.sizes = (
            sizes  # list of size_names and their corresponding area size in pixels
        )
        self.use_softmax = use_softmax
        self.binary = binary

        self.reset()

    def reset(self):
        self.reset_sized_metrics()
        self.reset_standard_metrics()
        self.reset_classwise_metrics()

    def reset_standard_metrics(self):
        self.tp_fp_tn_fn_counts = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    def reset_sized_metrics(self):
        self.tp_fp_tn_fn_counts_sized = {}
        for size in self.sizes:
            size_name, size_upper = size
            self.tp_fp_tn_fn_counts_sized[size_name] = {
                "TP": 0,
                "FN": 0,
            }
            
    def reset_classwise_metrics(self):
        self.tp_fp_tn_fn_counts_classwise = {}
        for el in self.classes:
            class_id = el[1]
            self.tp_fp_tn_fn_counts_classwise[class_id] = {
                "TP": 0,
                "FN": 0,
            }

    def compute_metrics(self):
        TP = self.tp_fp_tn_fn_counts["TP"]
        FP = self.tp_fp_tn_fn_counts["FP"]
        TN = self.tp_fp_tn_fn_counts["TN"]
        FN = self.tp_fp_tn_fn_counts["FN"]
        precision = 0 if TP + FP == 0 else TP / (TP + FP)
        recall = 0 if TP + FN == 0 else TP / (TP + FN)
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        if precision + recall == 0:
            f1 = 0
            f2 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            # f2 = (1+2^2) * (precision * recall) / (2^2 * precision + recall)

        return {
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "f1": f1,
        }

    def compute_sized_metrics(self):
        # TODO implement precision, recall, accuracy and f1 for each size
        return self.tp_fp_tn_fn_counts_sized
    
    def compute_classwise_metrics(self):
        # TODO implement precision, recall, accuracy and f1 for each class
        renamed = {el[0]: self.tp_fp_tn_fn_counts_classwise[el[1]] for el in self.classes}
        return renamed

    def bbox_area(self, bbox):
        # format x0,y0,x1,y1
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    def scale_bbox(self, x0, y0, x1, y1, scale_factor):
        x0_scaled = int(np.floor(x0 * scale_factor))
        y0_scaled = int(np.floor(y0 * scale_factor))
        x1_scaled = int(np.floor(x1 * scale_factor))
        y1_scaled = int(np.floor(y1 * scale_factor))
        return x0_scaled, y0_scaled, x1_scaled, y1_scaled

    def process_batch(self, logits_grid_batch, y_batch):
        prediction_errors = []
        for logits_grid, y in zip(logits_grid_batch, y_batch):
            logits_grid = logits_grid.cpu().detach()
            pred_errors = self.tp_fp_tn_fn(logits_grid, y)
            prediction_errors.append(pred_errors)
        return prediction_errors

    def get_size_name(self, bbox_area):
        for size in self.sizes:
            size_name, size_upper = size
            if size_upper is not None and bbox_area <= size_upper:
                return size_name
        return self.sizes[-1][0]

    def update_tp_fp_tn_fn_count(self, bbox_area, count_type: CountType, category_id=None):
        self.tp_fp_tn_fn_counts[count_type.value] += 1
        size = self.get_size_name(bbox_area)
        self.tp_fp_tn_fn_counts_sized[size][count_type.value] += 1
        # if category_id:
        #     self.tp_fp_tn_fn_counts_classwise[category_id][count_type.value] += 1
            
    def tp_fp_tn_fn(self, logits_grid, y_true):
        if self.use_softmax:
            preds_grid = torch.softmax(logits_grid, dim=0) > self.treshold
        else:
            preds_grid = logits_grid > self.treshold
        gt_grid = torch.zeros_like(preds_grid)
        
        has_FP = False
        has_FN = False
        
        if self.binary:
            preds_grid_new = torch.zeros((2, preds_grid.shape[1], preds_grid.shape[2]))
            preds_grid_new[0] = preds_grid[0]
            preds_grid_new[1] = torch.where(preds_grid.any(dim=0), 1, 0)
            preds_grid = preds_grid_new

        if "boxes" in y_true:
            for bbox, label in zip(y_true["boxes"], y_true["labels"]):
                bbox_area = self.bbox_area(bbox)
                x0, y0, x1, y1 = self.scale_bbox(*bbox.cpu().numpy(), self.scale_factor)
                if self.binary:
                    class_id = 1
                else:
                    class_id = label.item() + 1

                x0 = max(0, x0 - self.tolerance)
                y0 = max(0, y0 - self.tolerance)
                x1 = min(preds_grid.shape[2], x1 + self.tolerance)
                y1 = min(preds_grid.shape[1], y1 + self.tolerance)

                gt_grid[class_id, y0:y1, x0:x1] = 1
                if preds_grid[class_id, y0:y1, x0:x1].any():
                    self.update_tp_fp_tn_fn_count(bbox_area, CountType.TP, class_id - 1)
                else:
                    self.update_tp_fp_tn_fn_count(bbox_area, CountType.FN, class_id - 1)
                    has_FN = True

        # Calculating FP and TN for all classes excluding background
        for class_id in range(1, preds_grid.shape[0]):
            fp_mask = (preds_grid[class_id] == 1) & (gt_grid[class_id] == 0)
            tn_mask = (preds_grid[class_id] == 0) & (gt_grid[class_id] == 0)
            FP_count = fp_mask.sum().item()
            if FP_count > 0:
                has_FP = True
            self.tp_fp_tn_fn_counts["FP"] += FP_count
            self.tp_fp_tn_fn_counts["TN"] += tn_mask.sum().item()
            
        return {
            "has_FP": has_FP,
            "has_FN": has_FN,
        }