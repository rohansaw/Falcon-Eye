import numpy as np
import torch
from attention.pos_encoding import PositionalEncoding, PositionalEncodingOld
import lightning as L
import torch.nn.functional as F
from utils.metrics import Metrics
from utils.losses import loss_bce, loss_focal, loss_dist_bce, loss_dist_focal
from attention.mqa import MultiQuerySelfAttention
from attention.mhsa import MultiHeadedSelfAttention
#from attention.mqa_mn4 import MultiQueryAttention
from attention.mqa_fixed import MultiQueryAttention
from attention.sim_am import SimAM
import time
import wandb
import torch.nn as nn
from heads.altitude_modulation import AltitudeModulation

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class Detector(L.LightningModule):
    def __init__(
        self,
        backend,
        head,
        pos_weight,
        learning_rate,
        scale_factor,
        loss_type="bce",
        attention=None,
    ):
        super(Detector, self).__init__()
        self.save_hyperparameters(ignore=("backend", "head"))

        self.backend = backend
        self.head = head
        self.attention = attention
        self.positional_encoding = None
        self.pos_weight = pos_weight
        self.learning_rate = learning_rate
        self.loss_type = loss_type

        self.metrics = Metrics(
            scale_factor=scale_factor, tolerance=4, treshold=0.1, classes=[]
        )

        self.metrics_train = Metrics(
            scale_factor=scale_factor, tolerance=4, treshold=0.1, classes=[]
        )
        
        #self.channel_reducer = nn.Conv2d(128, 32, kernel_size=1)
        self.channel_reducer = nn.Identity()
        #self.altitude_modulation = AltitudeModulation(self.head.in_channels)

    def apply_attention(self, x):
        if isinstance(self.attention, MultiQuerySelfAttention) or isinstance(
            self.attention, MultiHeadedSelfAttention
        ) or isinstance(self.attention, nn.MultiheadAttention) or isinstance(self.attention, MultiQueryAttention):           
            x = self.channel_reducer(x)
            
            b, c, h, w = x.shape
            #if not self.positional_encoding:
            self.positional_encoding = PositionalEncoding(
                size=h, channels=c, device=self.device
            )
            b, c, h, w = x.shape
            x_features = x
            x_encoded = self.positional_encoding(x)
            x_encoded = x_encoded.flatten(start_dim=2).transpose(
                1, 2
            )  # (B, C, H, W) -> (B, H*W, C)
            
            if isinstance(self.attention, nn.MultiheadAttention):
                x_attention, _ = self.attention(x_encoded, x_encoded, x_encoded)
            else:
                x_attention = self.attention(x_encoded)
            x_attention = x_attention.transpose(1, 2)  # (batch_size, embed_dim, H*W)
            x_attention = x_attention.view(b, c, h, w) 
            x = x_features + x_attention
        elif isinstance(self.attention, SimAM):
            x = x + self.attention(x)
        else:
            raise ValueError(f"Attention {self.attention} not implemented")
        return x

    #def forward(self, x):
    def forward(self, x, altitudes=None):
        #t = time.perf_counter()
        x = self.backend(x)
        #print(f"Backend time: {time.perf_counter() - t}")
        
        if self.attention is not None:
            x = self.apply_attention(x)
        
        #print(altitudes)
        alt_norm = (altitudes - 1) / (800 - 1)
        # # # normalized_altitudes = alt_norm.view(-1, 1)
        # # # x = self.altitude_modulation(x, normalized_altitudes)
        
        # # # # # # #print(alt_norm)
        normalized_altitudes = alt_norm.view(-1, 1, 1, 1)  # Shape: (batch_size, 1, 1, 1)
        altitudes_map = normalized_altitudes.expand(-1, 1, x.shape[2], x.shape[3])
        x = torch.cat([x, altitudes_map], dim=1)
        
        mean = x.mean(dim=(0, 2, 3))  # shape: (3,)
        std = x.std(dim=(0, 2, 3))    # shape: (3,)

        print("Mean of each feature map:", mean)
        print("Standard deviation of each feature map:", std)
        
        combined_features_np = x.cpu().numpy()

        # Flatten each feature map for histogram plotting
        flattened_features = combined_features_np.reshape(combined_features_np.shape[0], combined_features_np.shape[1], -1)

        # Plot histograms for each feature map
        for i in range(flattened_features.shape[1]):
            plt.figure(figsize=(6, 4))
            plt.hist(flattened_features[:, i, :].flatten(), bins=50)
            plt.title(f'Feature Map {i} Distribution')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.savefig(f'feature_map_{i}_histogram.png')

        # For box plots, further reshape to (N*64*64, 3)
        reshaped_features = combined_features_np.reshape(-1, combined_features_np.shape[1])

        plt.figure(figsize=(15, 10))
        sns.boxplot(data=pd.DataFrame(reshaped_features, columns=[f'Feature Map {i}' for i in range(combined_features_np.shape[1])]))
        plt.xticks(rotation=90)
        plt.title('Box Plot of Feature Maps')
        plt.savefig('feature_maps_boxplot.png')
       
        #t = time.perf_counter()
        
        x = self.head(x)
        #print(f"Head time: {time.perf_counter() - t}")

        # if not self.training:
        #     assert len(x.shape) == 4
        #     x = torch.softmax(x, dim=1)
        return x

    def loss_fn(self, logits, y, distances=None):
        if self.loss_type == "bce":
            loss = loss_bce(logits, y, self.pos_weight)
        elif self.loss_type == "focal":
            loss = loss_focal(logits, y, self.pos_weight)
        elif self.loss_type == "dist_bce":
            loss = loss_dist_bce(logits, y, self.pos_weight, distances)
        elif self.loss_type == "dist_focal":
            loss = loss_dist_focal(logits, y, self.pos_weight, distances)
        else:
            raise ValueError(f"Loss {self.loss_type} not implemented")
        return loss
    
    def log_images(self, x, y):
        for img, labels in zip(x, y):
            all_boxes = []
            img_h, img_w = img.shape[1], img.shape[2]
            if "boxes" in labels:
                for bbox, label in zip(labels["boxes"], labels["labels"]):
                    x0, y0, x1, y1 = bbox
                    # get coordinates and labels
                    box_data = {"position" : {
                        "minX" : int(x0) / img_h,
                        "maxX" : int(x1) / img_h,
                        "minY" : int(y0) / img_w,
                        "maxY" : int(y1) / img_w,},
                        "class_id" : int(label),
                    }
                    all_boxes.append(box_data)
            box_image = wandb.Image(img, boxes = {"predictions": {"box_data": all_boxes}})
            # log all images and labels from batch to wandb to be visualized there
            wandb.log({"images_original": box_image})

    def training_step(self, batch, batch_idx):
        samples, y = batch
        
        x = [item[0] for item in samples]
        altitudes = [item[1] for item in samples]
        
        #t = time.perf_counter()
        train_segmentation_grids = torch.tensor(
            np.stack([el["train_segmentation_grid"] for el in y]), device=self.device)
        x = torch.stack(x)
        altitudes = torch.tensor(altitudes, dtype=torch.float32, device=x.device)
        #print(f"Data preparation time: {time.perf_counter() - t}")
        #self.log_images(x, [el["original_annotations"] for el in y])
        #t = time.perf_counter()
        logits = self(x, altitudes)
        #logits = self(x)
        #print(f"Forward time: {time.perf_counter() - t}")

        if self.loss_type == "dist_bce" or self.loss_type == "dist_focal":
            t = time.perf_counter()
            #distances = torch.stack([el["train_distance_penalty_grid"] for el in y])
            distances = torch.tensor(
                np.stack([el["train_distance_penalty_grid"] for el in y]),
                device=self.device
            )
            #print(f"Distance preparation time: {time.perf_counter() - t}")
            #t = time.perf_counter()
            loss = self.loss_fn(logits, train_segmentation_grids, distances)
            #print(f"Loss time: {time.perf_counter() - t}")
        else:
            loss = self.loss_fn(logits, train_segmentation_grids)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        original_annotations = [el["original_annotations"] for el in y]
        #t = time.perf_counter()
        self.metrics_train.process_batch(logits, original_annotations)
        #print(f"Metrics time: {time.perf_counter() - t}")
        return loss

    def validation_step(self, batch, batch_idx):
        samples, y = batch
        x = [item[0] for item in samples]
        altitudes = [item[1] for item in samples]
        x = torch.stack(x)
        altitudes = torch.tensor(altitudes, dtype=torch.float32, device=x.device)
        original_annotations = [el["original_annotations"] for el in y]
        logits = self(x, altitudes)
        
        centroid_grids = torch.tensor(
            np.stack([el["train_segmentation_grid"] for el in y]),
            device=self.device
        )
        
        if self.loss_type == "dist_bce" or self.loss_type == "dist_focal":
            distances = torch.tensor(
                np.stack([el["train_distance_penalty_grid"] for el in y]),
                device=self.device
            )
            loss = self.loss_fn(logits, centroid_grids, distances)
        else:
            loss = self.loss_fn(logits, centroid_grids)
        
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        self.metrics.process_batch(logits, original_annotations)

    def on_train_epoch_start(self) -> None:
        self.metrics_train.reset()

    def on_train_epoch_end(self) -> None:
        metrics = self.metrics_train.compute_metrics()
        metrics_by_size = self.metrics_train.compute_sized_metrics()
        metrics = {f"train_{k}": v for k, v in metrics.items()}
        print(metrics)
        print(metrics_by_size)
        self.log_dict(metrics, prog_bar=True)

    def on_validation_epoch_start(self) -> None:
        self.metrics.reset()

    def on_validation_epoch_end(self) -> None:
        metrics = self.metrics.compute_metrics()
        metrics_by_size = self.metrics.compute_sized_metrics()
        metrics = {f"val_{k}": v for k, v in metrics.items()}
        print(metrics)
        print(metrics_by_size)
        self.log_dict(metrics, prog_bar=True)
        for size in metrics_by_size:
            for k in metrics_by_size[size]:
                self.log(f"{k}_{size}", metrics_by_size[size][k], prog_bar=True)
            if metrics_by_size[size]["TP"] + metrics_by_size[size]["FN"] == 0:
                self.log(f"val_recall_{size}", 0)
            else:
                self.log(f"val_recall_{size}", metrics_by_size[size]["TP"] / (metrics_by_size[size]["TP"] + metrics_by_size[size]["FN"]))
            self.log_dict(metrics_by_size[size], prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]
