import torch
from typing import Tuple
import numpy as np
from torchvision.transforms import v2
import os
from PIL import Image, ImageDraw
import math

from torchvision import datasets, tv_tensors
from torchvision.transforms.v2 import functional as F
from collections import defaultdict
# from torchvision.tv_tensors._dataset_wrapper import WRAPPER_FACTORIES


def get_validation_transforms(
    img_size: Tuple[int, int],
    n_classes: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    scale_factor: float,
):
    return v2.Compose(
        [
            v2.Resize(img_size),
            v2.ToTensor(),
            v2.Normalize(mean=mean, std=std),
            ToGrid(n_classes, scale_factor),
        ]
    )


def collate_fn(batch):
    return tuple(zip(*batch))


class ToGrid(torch.nn.Module):
    def __init__(self, n_classes: int, scale_factor: int):
        super(ToGrid, self).__init__()
        self.n_classes = n_classes + 1  # including background
        self.grid_size = None
        self.scale_factor = scale_factor
        self.cache = {}

    def to_train_segmentation_grid_centerpoint(self, y):
        num_classes = self.n_classes  # including background
        map_h = self.grid_size[0]
        map_w = self.grid_size[1]
        grid = np.zeros(
            (num_classes, int(map_h), int(map_w))
        )  # Adjusted the shape to (num_classes, h, w)
        grid[0, :, :] = (
            1  # Set all feature map cells to background initially, assuming 0 is the background class
        )

        if "boxes" not in y:
            return grid
        for bbox, label in zip(y["boxes"], y["labels"]):
            x0 = bbox[0]
            y0 = bbox[1]
            x1 = bbox[2]
            y1 = bbox[3]

            x0_scaled = int(np.floor(x0 * self.scale_factor))
            y0_scaled = int(np.floor(y0 * self.scale_factor))
            x1_scaled = int(np.floor(x1 * self.scale_factor))
            y1_scaled = int(np.floor(y1 * self.scale_factor))

            bbox_map_centroid_x = int(np.floor((x0_scaled + x1_scaled) / 2))
            bbox_map_centroid_y = int(np.floor((y0_scaled + y1_scaled) / 2))

            # ensure centroids are within the grid
            bbox_map_centroid_x = min(max(0, bbox_map_centroid_x), map_w - 1)
            bbox_map_centroid_y = min(max(0, bbox_map_centroid_y), map_h - 1)

            class_id = (
                label + 1
            )  # Adjust the class ID to account for the background class
            grid[0, bbox_map_centroid_y, bbox_map_centroid_x] = (
                0 
            )
            grid[class_id, bbox_map_centroid_y, bbox_map_centroid_x] = (
                1  # Set the centroid pixel to the class ID
            )

        return grid

    def to_val_segmentation_grid(self, y):
        num_classes = self.n_classes  # including background
        map_h = self.grid_size[0]
        map_w = self.grid_size[1]
        grid = np.zeros(
            (num_classes, int(map_h), int(map_w))
        )  # Adjusted the shape to (num_classes, h, w)
        grid[0, :, :] = (
            1  # Set all feature map cells to background initially, 0 is the background class
        )

        if "boxes" not in y:
            return grid
        for bbox, label in zip(y["boxes"], y["labels"]):
            x0 = bbox[0]
            y0 = bbox[1]
            x1 = bbox[2]
            y1 = bbox[3]

            x0_scaled = int(np.floor(x0 * self.scale_factor))
            y0_scaled = int(np.floor(y0 * self.scale_factor))
            x1_scaled = int(np.floor(x1 * self.scale_factor))
            y1_scaled = int(np.floor(y1 * self.scale_factor))

            class_id = (
                label + 1
            )  # Adjust the class ID to account for the background class
            grid[0, y0_scaled:y1_scaled, x0_scaled:x1_scaled] = 0  # Set bakcground to 0
            grid[class_id, y0_scaled:y1_scaled, x0_scaled:x1_scaled] = 1

        return grid
    
    # def to_distances_grid(self, y):
    #     num_classes = self.n_classes  # including background
    #     map_h = self.grid_size[0]
    #     map_w = self.grid_size[1]
    #     grid = np.zeros(
    #         (num_classes, map_h, map_w)
    #     )  # initialize with 1 for easier background handling

    #     if "boxes" not in y:
    #         return grid

    #     # Prepare meshgrid for distance computation
    #     x_coords, y_coords = np.meshgrid(np.arange(map_w), np.arange(map_h))

    #     for bbox, label in zip(y["boxes"], y["labels"]):
    #         x0, y0, x1, y1 = bbox
    #         x0_scaled = int(np.floor(x0 * self.scale_factor))
    #         y0_scaled = int(np.floor(y0 * self.scale_factor))
    #         x1_scaled = int(np.floor(x1 * self.scale_factor))
    #         y1_scaled = int(np.floor(y1 * self.scale_factor))

    #         # Compute centroid
    #         centroid_x = (x0_scaled + x1_scaled) // 2
    #         centroid_y = (y0_scaled + y1_scaled) // 2
            
    #         centroid_x = min(max(0, centroid_x), map_w - 1)
    #         centroid_y = min(max(0, centroid_y), map_h - 1)

    #         # Compute maximum distance inside the bounding box for normalization
    #         max_distance = np.sqrt(
    #             ((x1_scaled - x0_scaled) / 2) ** 2 + ((y1_scaled - y0_scaled) / 2) ** 2
    #         )
            
    #         if max_distance == 0:
    #             max_distance = 1
                
    #         # Create masks for bounding box area to limit computations
    #         # x_mask = (x_coords >= x0_scaled) & (x_coords <= x1_scaled)
    #         # y_mask = (y_coords >= y0_scaled) & (y_coords <= y1_scaled)
    #         # bbox_mask = x_mask & y_mask

    #         # Calculate distances from centroid within the bounding box
    #         distances = np.sqrt(
    #             (x_coords - centroid_x) ** 2 + (y_coords - centroid_y) ** 2
    #         )
            

    #         # Update the grid for this label using minimum distance
    #         class_id = label + 1
    #         grid[class_id] = normalized_distances

    #         # Update background distances
    #         grid[0] = normalized_distances
    #         # TODO: This only works if there is a single class, otherwise we have to take min or max?? over the classes from before

    #     return grid


    def to_dist_penalty_train_grid(self, y):
        num_classes = self.n_classes  # including background
        map_h = self.grid_size[0]
        map_w = self.grid_size[1]
        grid = np.zeros((num_classes, map_h, map_w))  # initialize with 1 for no penalty

        if "boxes" not in y:
            return grid

        # Prepare meshgrid for distance computation
        x_coords, y_coords = np.meshgrid(np.arange(map_w), np.arange(map_h))

        for bbox, label in zip(y["boxes"], y["labels"]):
            x0, y0, x1, y1 = bbox
            x0_scaled = int(np.floor(x0 * self.scale_factor))
            y0_scaled = int(np.floor(y0 * self.scale_factor))
            x1_scaled = int(np.floor(x1 * self.scale_factor))
            y1_scaled = int(np.floor(y1 * self.scale_factor))

            # Compute centroid
            centroid_x = (x0_scaled + x1_scaled) // 2
            centroid_y = (y0_scaled + y1_scaled) // 2
            
            centroid_x = min(max(0, centroid_x), map_w - 1)
            centroid_y = min(max(0, centroid_y), map_h - 1)

            # Compute maximum distance inside the bounding box for normalization
            max_distance = np.sqrt(
                ((x1_scaled - x0_scaled) / 2) ** 2 + ((y1_scaled - y0_scaled) / 2) ** 2
            )
            
            if max_distance == 0:
                max_distance = 1
                
            # Calculate distances from centroid within the bounding box
            distances = np.sqrt(
                (x_coords - centroid_x) ** 2 + (y_coords - centroid_y) ** 2
            )
            
            sigma = 1.5
            gaussian_distances = 1 - np.exp(-(distances[y0_scaled:y1_scaled, x0_scaled:x1_scaled] ** 2) / (2 * sigma ** 2))
            # print(gaussian_distances.shape)
            # print(gaussian_distances)
            # exit()
            
            normalized_distances = np.clip(distances / max_distance, 0, 1)
            # new_min = 0.5
            # new_max = 1
            # normalized_distances = normalized_distances * (new_max - new_min) + new_min
            
            
            # Invert distances to weigh errors close to the center less
            # penalty_weights = np.zeros((map_h, map_w))
            # penalty_weights[y0_scaled:y1_scaled, x0_scaled:x1_scaled] = gaussian_distances
            penalty_weights = normalized_distances
            penalty_weights[centroid_y, centroid_x] = 1
            # print(normalized_distances[y0_scaled:y1_scaled, x0_scaled:x1_scaled])
            # exit()
            
            # adjsut the penalty weight to be atleast 0.5 for the pixels closest to the center
            # this should be done by mainting the relationships between the distacnes of the pixels

            # Update the grid for this label using maximum weights (to handle multiple objects)
            class_id = label + 1
            
            # compute the mask for the bounding box (x0_scaled, y0_scaled) to (x1_scaled, y1_scaled)
            bbox_grid_mask = np.zeros((map_h, map_w))
            bbox_grid_mask[y0_scaled:y1_scaled, x0_scaled:x1_scaled] = 1
            
            # ToDo use bbox grid mask to update the grid, by only taking the values in this area and updating them without max
            
            
            # update the grid in the bounding box area with the penalty weights
            grid[class_id] = np.maximum(grid[class_id], penalty_weights * bbox_grid_mask)
            grid[0] = np.maximum(grid[0], penalty_weights * bbox_grid_mask)
            
            # grid[class_id] = np.maximum(grid[class_id], penalty_weights)

            # # Update background distances
            # grid[0] = np.maximum(grid[0], penalty_weights)
            
        # update all zeros with 1
        grid[grid == 0] = 1
            
        return grid
    
    

    def forward(self, img, labels):
        # need to set this here to account for padding from transformations etc
        self.grid_size = int(math.ceil(img.shape[1] * self.scale_factor)), int(math.ceil(img.shape[2] * self.scale_factor))
        labels_new = {
            "train_segmentation_grid": self.to_train_segmentation_grid_centerpoint(labels),
            "train_distance_penalty_grid": self.to_dist_penalty_train_grid(labels),
            "val_segmentation_grid": self.to_val_segmentation_grid(labels),
            "original_annotations": labels,
        }
        
        # save training segmentation grid and distance penalty grid as images for class 
        
        # check that at class 1 any value is not 0
        # if np.any(labels_new["train_segmentation_grid"][1] != 0):
        #     out_folder = "output"
        #     os.makedirs(out_folder, exist_ok=True)
        #     np.savetxt("output/train_seg_grid", labels_new["train_segmentation_grid"][1])
        #     np.savetxt("output/distances", labels_new["train_distance_penalty_grid"][1])
            
        #     # save img which is a pytorch tensor as an image on disk
        #     img = img.permute(1, 2, 0).numpy()
        #     img = (img * 255).astype(np.uint8)
        #     img = Image.fromarray(img)
        #     img.save(os.path.join(out_folder, "img.png"))
            
        #     img_w, img_h = labels_new["train_segmentation_grid"][1].shape
        #     # create a white image
        #     img = Image.new("RGB", (img_w, img_h), color="black")
        #     draw = ImageDraw.Draw(img)
        #     grid = labels_new["train_segmentation_grid"][1]
        #     grid = np.uint8(grid * 255)
        #     grid = Image.fromarray(grid)
        #     #img.paste(grid)
        #     grid.save(os.path.join(out_folder, f"train_segmentation_grid_{1}.png"))
        #     grid = labels_new["train_distance_penalty_grid"][1]
        #     grid = np.uint8(255 - (grid * 255))
        #     grid = Image.fromarray(grid)
        #     draw.text((0, 0), f"Class {1}", fill="white")
        #     #img.paste(grid, (0, 20))
        #     grid.save(os.path.join(out_folder, f"train_distance_penalty_grid_{1}.png"))
            
        #     exit()

        return img, labels_new


class DynamicScaling(torch.nn.Module):
    def __init__(self, target_gsd: float):
        super(DynamicScaling, self).__init__()
        self.target_gsd = target_gsd

    def forward(self, img, meta, labels):
        img_w, img_h = img.size
        altitude = meta["altitude"]
        focal_len_mm = (
            meta["focal_length"] if meta["focal_length"] is not None else 3.04
        )  # PiCamV2 specifics
        sensor_width_mm = 3.68
        sensor_width_px = 3280  # ToDo make changable
        focal_len_px = focal_len_mm * (sensor_width_px / sensor_width_mm)
        img_size_w = (altitude / focal_len_px * img_w) / self.target_gsd
        img_size_h = (altitude / focal_len_px * img_h) / self.target_gsd

        scale_factor_w = img_size_w / img_w
        scale_factor_h = img_size_h / img_h

        # save image before with drawn bounding box
        out_folder = "output"
        os.makedirs(out_folder, exist_ok=True)
        draw = ImageDraw.Draw(img)

        # Draw original bounding boxes
        for box in labels["boxes"]:
            # The box is expected to be in the format [xmin, ymin, xmax, ymax]
            draw.rectangle(box.tolist(), outline="red", width=3)
        img.save(os.path.join(out_folder, "before_resizing2.png"))
        print("saved 1")

        img = img.resize((int(img_size_w), int(img_size_h)))
        print("resized")
        # Scale the bounding boxes
        labels["boxes"] = labels["boxes"] * torch.tensor(
            [scale_factor_w, scale_factor_h, scale_factor_w, scale_factor_h]
        )
        print("processed labels")

        draw = ImageDraw.Draw(img)
        print("draw")
        for box in labels["boxes"]:
            draw.rectangle(box.tolist(), outline="red", width=3)
        print("finish draw")

        # Save image after resizing with drawn bounding box
        img.save(os.path.join(out_folder, "after_resizing2.png"))
        print("finish shave")

        if altitude < 300:
            exit()

        return img, labels

class ImageTransformWrapper:
    def __init__(self, transform, include_metadata=True):
        self.transform = transform
        self.include_metadata = include_metadata

    def __call__(self, img, labels):
        x, metadata = img
        x, labels = self.transform(x, labels)
        if self.include_metadata:
            return (x, metadata), labels
        return x, labels


def parse_target_keys(target_keys, *, available, default):
    if target_keys is None:
        target_keys = default
    if target_keys == "all":
        target_keys = available
    else:
        target_keys = set(target_keys)
        extra = target_keys - available
        if extra:
            raise ValueError(f"Target keys {sorted(extra)} are not available")

    return target_keys

def list_of_dicts_to_dict_of_lists(list_of_dicts):
    dict_of_lists = defaultdict(list)
    for dct in list_of_dicts:
        for key, value in dct.items():
            dict_of_lists[key].append(value)
    return dict(dict_of_lists)

