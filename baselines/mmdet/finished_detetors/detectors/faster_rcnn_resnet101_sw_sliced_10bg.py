_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_sw_detection_sliced_10bg.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

work_dir = f"/data/results/faster_rcnn_resnet101/{_base_.ds_name}_10bg"
num_classes = len(_base_.classes)


train_cfg = dict(
    max_epochs=50,
)

model = dict(
    data_preprocessor={{_base_.data_preprocessor}},
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')),
    #rpn_head=dict(
    #    reg_decoded_bbox=True,
    #    loss_bbox=dict(type='CIoULoss', loss_weight=1.0)),
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64])
    ),
    roi_head=dict(
        bbox_head=dict(
            #reg_decoded_bbox=True,
            #loss_bbox=dict(type='CIoULoss', loss_weight=1.0),
            num_classes=num_classes)
    )
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001, nesterov=True))



default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        save_best="coco/bbox_mAP",
        rule="greater",
        max_keep_ckpts=5
    ),
    early_stopping=dict(
        type="EarlyStoppingHook",
        monitor="coco/bbox_mAP",
        patience=10,
        min_delta=0.005)
)

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='WandbVisBackend',
                    init_kwargs={
                        'project': 'mmdetection',
                        'group': f'frcnn_{_base_.ds_name}'
                    })]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')