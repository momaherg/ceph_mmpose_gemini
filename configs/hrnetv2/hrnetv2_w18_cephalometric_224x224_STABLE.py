# Inherit from default runtime settings
_base_ = ['../_base_/default_runtime.py']

# Import custom dataset meta-information
from cephalometric_dataset_info import dataset_info as cephalometric_metainfo

# Model settings
pretrained_model = 'https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth'

# Input image size
input_size = (224, 224)

# Number of keypoints
num_keypoints = 19

# Model definition
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True
    ),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(18, 36)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144),
                multiscale_output=True),
            upsample=dict(mode='bilinear', align_corners=False)),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained_model)
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=[18, 36, 72, 144],
        input_transform='resize_concat',
        out_channels=num_keypoints,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=dict(
            type='MegviiHeatmap',
            input_size=input_size,
            heatmap_size=(input_size[0] // 4, input_size[1] // 4),
            kernel_size=7,
        )
    ),
    test_cfg=dict(
        flip_test=False,
        flip_mode='heatmap',
        shift_heatmap=True,
    )
)

# Dataset settings
dataset_type = 'CustomCephalometricDataset'
data_root = '/content/drive/MyDrive/Lala\'s Masters/'

# Pipelines
common_pipeline_prefix = [
    dict(type='LoadImageNumpy'),
    dict(type='GetBBoxCenterScale'),
]

common_pipeline_suffix = [
    dict(type='TopdownAffine', input_size=input_size),
    dict(type='GenerateTarget', 
         encoder=dict(
            type='MSRAHeatmap',
            input_size=input_size, 
            heatmap_size=(input_size[0] // 4, input_size[1] // 4),
            sigma=3.0  # INCREASED from 2 to 3 for better target coverage
        )
    ),
    dict(type='CustomPackPoseInputs', 
         meta_keys=(
             'img_id', 'img_path', 'ori_shape', 'img_shape',
             'input_size', 'input_center', 'input_scale',
             'flip', 'flip_direction',
             'num_joints', 'joint_weights',
             'id', 'patient_text_id', 'set', 'class'
            )
        )
]

val_test_pipeline_suffix = [
    dict(type='TopdownAffine', input_size=input_size),
    dict(type='CustomPackPoseInputs', 
         meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape',
            'input_size', 'input_center', 'input_scale',
            'flip', 'flip_direction',
            'num_joints', 'joint_weights',
            'id', 'patient_text_id', 'set', 'class'
            )
        )
]

# REDUCED augmentation to prevent overfitting
train_pipeline = [
    *common_pipeline_prefix,
    dict(type='RandomBBoxTransform',
         scale_factor=[0.9, 1.1],  # REDUCED from [0.8, 1.2]
         rotate_factor=10,         # REDUCED from 15
         shift_factor=0.0),
    *common_pipeline_suffix
]

val_pipeline = [
    *common_pipeline_prefix,
    *val_test_pipeline_suffix
]

test_pipeline = val_pipeline

# Dataloaders - CRITICAL FIXES
train_dataloader = dict(
    batch_size=8,  # REDUCED from 32 to 8 for stable gradients
    num_workers=2,  # REDUCED to avoid bottlenecks
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_data_pure_old_numpy.json',
        metainfo=cephalometric_metainfo,
        pipeline=train_pipeline,
        test_mode=False,
    )
)

# PROPER VALIDATION SETUP
val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CustomCephalometricDataset',
        data_root=data_root,
        ann_file='train_data_pure_old_numpy.json',
        metainfo=cephalometric_metainfo,
        pipeline=val_pipeline,
        test_mode=True,
    )
)

test_dataloader = None

# Evaluator
val_evaluator = dict(
    type='PCKAccuracy',
    thr=0.05,
    norm_item=['bbox', 'torso']
)
test_evaluator = None

# CONSERVATIVE TRAINING SCHEDULE
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=2)  # FURTHER REDUCED epochs for safety
val_cfg = dict()
test_cfg = None

# AGGRESSIVE LEARNING RATE REDUCTION
optim_wrapper = dict(optimizer=dict(
    type='AdamW',
    lr=2e-5,  # DRASTICALLY REDUCED from 1e-4 to 2e-5
    weight_decay=0.01  # INCREASED regularization from 0.0001
))

# VERY GENTLE SCHEDULER
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,  # HIGHER start factor for more conservative warmup
        by_epoch=False,
        begin=0,
        end=200  # REDUCED warmup iterations
    ),
    dict(
        type='CosineAnnealingLR',
        begin=0, 
        end=20,  # Match reduced max_epochs
        by_epoch=True,
        T_max=20,
        eta_min=1e-7  # Even lower minimum LR
    )
]

# ENHANCED MONITORING HOOKS
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),  # More frequent logging
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=1, 
        save_best='PCK', 
        rule='greater',
        max_keep_ckpts=5  # Keep more checkpoints to find best
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False)
)

# NO CUSTOM HOOKS - manual monitoring instead
# Monitor training manually and stop when validation PCK plateaus 