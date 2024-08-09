# dataset settings
import os
dataset_type ="CocoDataset"
classes= ('boat',)
ds_name = "sw_sliced"
data_root =  "/data/" + ds_name + "/coco"
train_prefix = "train"
val_prefix = "val"
train_json = 'annotations/instances_train-50percent_bg.json'
val_json = 'annotations/instances_val-50percent_bg.json'
backend_args = None
mean = [i*255 for i in [0.2761, 0.4251, 0.5644]]
std = [i*255 for i in [0.2060, 0.1864, 0.2218]]
img_size = (512,512)
batch_size = 16

data_preprocessor = dict(
    type='mmdet.DetDataPreprocessor', mean=mean, std=std, bgr_to_rgb=True,
    pad_size_divisor=32
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_size, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=20,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=18,
    ),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_size, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=train_json,
        data_prefix=dict(img=train_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        data_prefix=dict(img=val_prefix),
        ann_file=val_json,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    outfile_prefix=f"/data/results/ssd_lite/{ds_name}",
    ann_file=os.path.join(data_root, val_json),
    metric='bbox',
    proposal_nums=(1, 10, 100, 300, 1000),
    format_only=False,
    #use_mp_eval=True,
    classwise=True,
    backend_args=backend_args)

test_evaluator = val_evaluator