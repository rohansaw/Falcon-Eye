import lightning as L
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import os
import numpy as np
from typing import Tuple
import multiprocessing


from torchvision.transforms import v2
from dataset.transforms import ImageTransformWrapper, ToGrid, DynamicScaling, collate_fn, get_validation_transforms
from dataset.v2_wrapper import wrap_dataset_for_transforms_v2

from dataset.sliced_coco_dataset import SlicedCocoDataset
from dataset.mosaic_augmentation import MosaicDataset


class DataModuleCOCO(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        img_size: Tuple[int, int],
        grid_shape: Tuple[int, int, int],
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        num_workers: int = 8,
        ann_file_train: str = None,
        ann_file_val: str = None,
        scale_factor: float = 0.125,
        slicing_config_train: dict = None,
        slicing_config_val: dict = None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.ann_file_train = ann_file_train
        self.ann_file_val = ann_file_val

        p = 0.5
        self.num_workers = num_workers
        self.mean = mean
        self.std = std
        self.grid_shape = grid_shape
        self.scale_factor = scale_factor
        self.slicing_config_train = slicing_config_train
        self.slicing_config_val = slicing_config_val
        
        # manager = multiprocessing.Manager()
        # self.shared_tiles_array = manager.list()
        
        num_classes = grid_shape[0] - 1

        required_padding = int((np.sqrt(self.img_size[0] ** 2 + self.img_size[1] ** 2) - np.min(self.img_size)) / 2)
        padded_img_size = (self.img_size[0] + 2 * required_padding, self.img_size[1] + 2 * required_padding)
        self.transforms_train = ImageTransformWrapper(v2.Compose(
            [
                v2.Resize(self.img_size),
                v2.Pad(required_padding),
                v2.RandomHorizontalFlip(p),
                v2.RandomResizedCrop(padded_img_size, scale=(0.85, 1.15)),
                #v2.RandomResizedCrop(self.img_size, scale=(0.85, 1.15)),
                v2.RandomVerticalFlip(p),
                #v2.RandomPerspective(),
                v2.RandomRotation(180),
                v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.2),
                v2.ToTensor(),
                v2.Normalize(
                    mean=self.mean,
                    std=self.std,
                ),
                #v2.RandomErasing(p=0.25),
                ToGrid(num_classes, self.scale_factor),
            ]
        ))
        

        self.transforms_val = ImageTransformWrapper(get_validation_transforms(
            self.img_size, num_classes, mean=self.mean, std=self.std, scale_factor=self.scale_factor
        ))

    def setup(self, stage: str):
        data_dir_train = os.path.join(self.data_dir, "train")
        data_dir_val = os.path.join(self.data_dir, "val")
        ann_file_train = (
            self.ann_file_train
            if self.ann_file_train
            else os.path.join(self.data_dir, "annotations", "instances_train.json")
        )
        ann_file_val = (
            self.ann_file_val
            if self.ann_file_val
            else os.path.join(self.data_dir, "annotations", "instances_val.json")
        )
        
        if self.slicing_config_train:
            self.load_unsliced_datasets(data_dir_train, ann_file_train, data_dir_val, ann_file_val)
        else:
            self.load_presliced_dataset(data_dir_train, ann_file_train, data_dir_val, ann_file_val)
        
        self.coco_train = wrap_dataset_for_transforms_v2(self.coco_train)
        self.coco_val = wrap_dataset_for_transforms_v2(self.coco_val)
        print("Num samples train dataset: ", len(self.coco_train))
        print("Num samples val dataset: ", len(self.coco_val))
        
        
    def load_presliced_dataset(self, data_dir_train, ann_file_train, data_dir_val, ann_file_val):
        self.coco_train = CocoDetection(
            root=data_dir_train,
            annFile=ann_file_train,
            transforms=self.transforms_train,
        )
        self.coco_val = CocoDetection(
            root=data_dir_val, annFile=ann_file_val, transforms=self.transforms_val
        )
        
        
    def load_unsliced_datasets(self, data_dir_train, ann_file_train, data_dir_val, ann_file_val):
        self.coco_train = SlicedCocoDataset(
            tile_sz = self.slicing_config_train["tile_sz"],
            overlap = self.slicing_config_train["overlap"],
            min_area_ratio = self.slicing_config_train["min_area_ratio"],
            use_cache = self.slicing_config_train["use_cache"],
            target_gsd = self.slicing_config_train["target_gsd"],
            amount_bg = self.slicing_config_train["amount_bg"],
            full_image = self.slicing_config_train["full_image"],
            root=data_dir_train,
            annFile=ann_file_train,
            transforms=self.transforms_train,
            meta_keys=["altitude", "focal_length"],
            use_mosaic=self.slicing_config_train["use_mosaic"],
            #shared_tiles_array=self.shared_tiles_array,
        )

        self.coco_val = SlicedCocoDataset(
           tile_sz = self.slicing_config_val["tile_sz"],
            overlap = self.slicing_config_val["overlap"],
            min_area_ratio = self.slicing_config_val["min_area_ratio"],
            use_cache = self.slicing_config_val["use_cache"],
            target_gsd = self.slicing_config_val["target_gsd"],
            amount_bg = self.slicing_config_val["amount_bg"],
            full_image = self.slicing_config_val["full_image"],
            root=data_dir_val,
            annFile=ann_file_val,
            transforms=self.transforms_val,
            meta_keys=["altitude", "focal_length"]
        )


    def train_dataloader(self):
        return DataLoader(
            self.coco_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.coco_val,
            batch_size=self.batch_size,
            num_workers=int(self.num_workers/2),
            collate_fn=collate_fn,
            persistent_workers=True,
        )
