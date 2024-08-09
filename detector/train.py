from attention.mhsa import MultiHeadedSelfAttention
from attention.mqa import MultiQuerySelfAttention
from attention.sim_am import SimAM
from model import Detector
from heads.fcn_head import FCNHead
from backends.mobilenet_v2 import MobileNetV2
from backends.mobileone import mobileone
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from dataset.data_module_coco import DataModuleCOCO
from lightning.pytorch.callbacks import ModelCheckpoint
import os
import torch
import numpy as np
import random
import wandb
from utils.visualization import visualize_augmentation
from torch.nn import MultiheadAttention
#from attention.mqa_mn4 import MultiQueryAttention
from attention.mqa_fixed import MultiQueryAttention

import torch
from torch.profiler import profile, record_function, ProfilerActivity

# Ensure deterministic behavior
seed = 2203
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(batch_size, epochs, pos_weight, lr, bn_momentum, loss_type):
    num_classes = 1 # 12 for visdrone, 6 for SDS, 1 for SW
    input_shape = (3, 512, 512)
    
    # data_root = "/data/PeopleOnGrass"
    # ann_file_train = "/data/PeopleOnGrass/annotations/instances_train_bev.json"
    # ann_file_val = "/data/PeopleOnGrass/annotations/instances_val_bev.json"

    data_root = "/data/sw/coco/"
    ann_file_train = "/data/sw/coco/annotations/instances_train.json"
    ann_file_val = "/data/sw/coco/annotations/instances_val.json"
    
    # data_root = "/data/sds/coco/"
    # ann_file_train = "/data/sds/coco/annotations/instances_train.json"
    # ann_file_val = "/data/sds/coco/annotations/instances_val.json"
    model_out_root = "/data/results/my-detector"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # SW
    mean = [0.2761, 0.4251, 0.5644]
    std = [0.2060, 0.1864, 0.2218]
    
    # SDS
    # mean = [0.4263, 0.4856, 0.4507]
    # std = [0.1638, 0.1515, 0.1755]
    
    # PeopleOnGrass
    # mean = [0.43717304, 0.44310874, 0.33362516]
    # std = [0.23407398, 0.21981522, 0.2018422 ]
    
    # Visdrone
    # mean =[0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    
    # data_root = "/data/sw_sliced_centered/coco/"
    # ann_file_train = (
    #     "/data/sw_sliced_centered/coco/annotations/instances_train-50percent_bg.json"
    # )
    # ann_file_val = (
    #     "/data/sw_sliced_centered/coco/annotations/instances_val-0percent_bg.json"
    # )
    # data_root = "/data/sds_sliced_centered/coco/"
    # ann_file_train = "/data/sds_sliced_centered/coco/annotations/instances_train_cleaned.json"
    # ann_file_val = "/data/sds_sliced_centered/coco/annotations/instances_val_cleaned.json"
    
    # data_root = "/data/visdrone/"
    # ann_file_train = "/data/visdrone/annotations/instances_train.json"
    # ann_file_val = "/data/visdrone/annotations/instances_val.json"
    
    # data_root = "/data/sw_original/"
    # ann_file_train = "/data/sw_original/annotations/instances_train.json"
    # ann_file_val = "/data/sw_original/annotations/instances_val.json"

    #backend = MobileNetV2(bn_momentum=bn_momentum)

    backend = mobileone(variant="s0", bn_momentum=bn_momentum)
    checkpoint = torch.load(
        "pretrained_models/mobileone_s0_unfused.pth.tar",
        map_location=device,
    )
    backend.load_state_dict(checkpoint)
    backend.truncate(2)

    feat_map_shape = backend.get_feature_map_shape(input_shape)
    print(feat_map_shape)
    
    channels = feat_map_shape[0]
    # Add +1 if using altitude for classification
    
    head = FCNHead(
        num_classes=num_classes, in_channels=channels, middle_channels=32
    )
    scale_factor = feat_map_shape[1] / input_shape[1]
    attention = None

    # attention = MultiHeadedSelfAttention(
    #     dim_in=feat_map_shape[0], dim_k=int(feat_map_shape[0] / 8), num_heads=4
    # )

    # attention = MultiQueryAttention(
    #     d_model=feat_map_shape[0],
    #     heads=8,
    #     attn_impl="torch"
    # )

    attention = SimAM()
    
    
    #attention = MultiheadAttention(channels, 4, batch_first=True)
    
    #attention = MultiQueryAttention(channels, 4)
    
    print(channels)
    # attention = MultiQueryAttention(
    #     num_heads=4,
    #     key_dim=channels,
    #     value_dim=channels,
    # )

    model = Detector(
        head=head,
        backend=backend,
        pos_weight=pos_weight,
        learning_rate=lr,
        scale_factor=scale_factor,
        attention=attention,
        loss_type=loss_type,
    )
    
    # checkpoint_pretrained = "artifacts/model-x75rk874:v2/model.ckpt"
    # pretrained_head = head = FCNHead(
    #     num_classes=12, in_channels=feat_map_shape[0], middle_channels=32
    # )
    # pretrained_model =  (Detector.load_from_checkpoint(checkpoint_pretrained, head=pretrained_head,
    #     backend=backend)
    #     .type(torch.FloatTensor)
    # )
    
    # model.backend = pretrained_model.backend
    # model.backend.to(device)

    grid_shape = (
        num_classes + 1,
        feat_map_shape[1],
        feat_map_shape[2],
    )  # (num_classes + 1, height, width)
    img_size = input_shape[1:]
    
    slicing_config_train = {
        "tile_sz": 512,
        "overlap": 0.25,
        "min_area_ratio": 1,
        "use_cache": True,
        "target_gsd": None, #0.04, # 0.04 == 4cm/px
        "amount_bg": 0.15,
        "full_image": False,
        "use_mosaic": False,
    }
    
    slicing_config_val = {
        "tile_sz": 512,
        "overlap": 0.25,
        "min_area_ratio": 1,
        "use_cache": True,
        "target_gsd": None, #0.04, # 0.04 == 4cm/px
        "amount_bg": 0.15, # if > 1, all tiles are used, default 0.15
        "full_image": False,
        "use_mosaic": False,
    }
    
    datamodule = DataModuleCOCO(
        data_dir=data_root,
        ann_file_train=ann_file_train,
        ann_file_val=ann_file_val,
        batch_size=batch_size,
        img_size=img_size,
        grid_shape=grid_shape,
        mean=mean,
        std=std,
        num_workers=8,
        scale_factor=scale_factor,
        slicing_config_train=slicing_config_train,
        slicing_config_val=slicing_config_val,
    )
    
    # datamodule.setup(stage='fit')
    # visualize_augmentation(datamodule.coco_train, datamodule.transforms_train)

    wandb_logger = WandbLogger(log_model="all", project="my-detector")
    #wandb_logger.watch(model, log='all')
    config = {
        "data_dir": data_root,
        "ann_file_train": ann_file_train,
        "ann_file_val": ann_file_val,
        "batch_size": batch_size,
        "bn_momentum": bn_momentum,
        "epochs": epochs,
        "pos_weight": pos_weight,
        "lr": lr,
        }
    
    print("Using config: ", config)
    
    config.update(slicing_config_train)
    config.update(slicing_config_val)
    wandb_logger.experiment.config.update(config)

    model_save_dir = os.path.join(model_out_root, wandb_logger.experiment.id)
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_dir, monitor="val_f1", mode="max", save_top_k=5
    )
    
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=epochs,
        accelerator=device,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback],
    )
    
    #trainer.validate(ckpt_path="artifacts/model-9xtt9z6j:v66/model.ckpt", model=model, datamodule=datamodule)
    
    wandb.define_metric("val_f1", summary="max")
    wandb.define_metric("val_recall", summary="max")
    wandb.define_metric("val_precision", summary="max")
    trainer.fit(model=model, datamodule=datamodule)
    
def main(config=None):
    lr = 0.001 if config is None else config.lr
    pos_weight = 10 if config is None else config.pos_weight # Todo change to default 10
    batch_size = 32 if config is None else config.batch_size
    epochs = 300 if config is None else config.epochs
    bn_momentum = 0.9 if config is None else config.bn_momentum
    loss_type = "dist_bce" if config is None else config.loss_type
    
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  profile_memory=True, record_shapes=True) as prof:
    #     with record_function("model_inf"):
    train(batch_size=batch_size, epochs=epochs, pos_weight=pos_weight, lr=lr, bn_momentum=bn_momentum, loss_type=loss_type)
            
    #print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
       


if __name__ == "__main__":
    main()
