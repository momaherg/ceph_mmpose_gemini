# Inherit from default runtime settings
_base_ = ['../_base_/default_runtime.py']

# Import custom dataset meta-information
from cephalometric_dataset_info import dataset_info as cephalometric_metainfo

# Model settings
pretrained_model = 'https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth'
input_size = (224, 224)
num_keypoints = 19

# Codec settings (following AFLW pattern)
codec = dict(
    type='MSRAHeatmap', 
    input_size=input_size, 
    heatmap_size=(56, 56),  # 224//4 = 56
    sigma=2
)

# Model definition - Following AFLW architecture pattern
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
    # ADD NECK - This is crucial for proper feature processing
    neck=dict(
        type='FeatureMapProcessor',
        concat=True,  # Concatenate multi-scale features
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=270,  # 18+36+72+144 = 270 (concatenated features)
        out_channels=num_keypoints,
        deconv_out_channels=None,
        conv_out_channels=(270, ),  # Keep same channels
        conv_kernel_sizes=(1, ),    # 1x1 conv
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec
    ),
    test_cfg=dict(
        flip_test=False,  # Cephalometric landmarks are not symmetric
        flip_mode='heatmap',
        shift_heatmap=True,
    )
)

# Dataset settings
dataset_type = 'CustomCephalometricDataset'
data_root = '/content/drive/MyDrive/Lala\'s Masters/'

# Pipelines - Simplified following AFLW pattern
train_pipeline = [
    dict(type='LoadImageNumpy'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal', prob=0.0),  # Disabled for cephalometric
    dict(
        type='RandomBBoxTransform',
        shift_prob=0.3,  # Allow some shifting
        rotate_factor=15,  # Reduced rotation
        scale_factor=(0.9, 1.1)  # Less aggressive scaling
    ),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='CustomPackPoseInputs',
         meta_keys=(
             'img_id', 'img_path', 'ori_shape', 'img_shape',
             'input_size', 'input_center', 'input_scale',
             'flip', 'flip_direction',
             'num_joints', 'joint_weights',
             'id', 'patient_text_id', 'set', 'class'
         ))
]

val_pipeline = [
    dict(type='LoadImageNumpy'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='CustomPackPoseInputs',
         meta_keys=(
             'img_id', 'img_path', 'ori_shape', 'img_shape',
             'input_size', 'input_center', 'input_scale',
             'flip', 'flip_direction',
             'num_joints', 'joint_weights',
             'id', 'patient_text_id', 'set', 'class'
         ))
]

test_pipeline = val_pipeline

# Data loaders - Following AFLW pattern with auto-scaling
train_dataloader = dict(
    batch_size=16,  # Reduced for stability
    num_workers=4,
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

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_data_pure_old_numpy.json',
        metainfo=cephalometric_metainfo,
        pipeline=val_pipeline,
        test_mode=True,
    )
)

test_dataloader = None

# Training schedule
train_cfg = dict(max_epochs=60, val_interval=5)
val_cfg = dict()
test_cfg = None

# Optimizer - Following AFLW pattern with Adam and higher LR
optim_wrapper = dict(optimizer=dict(
    type='Adam',  # Changed from AdamW to Adam
    lr=5e-4,      # Increased from 1e-4 but lower than AFLW's 2e-3
))

# Learning rate scheduler - Back to MultiStepLR like AFLW
param_scheduler = [
    dict(
        type='LinearLR', 
        begin=0, 
        end=500, 
        start_factor=0.001,
        by_epoch=False  # Warmup
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=60,
        milestones=[30, 50],  # Earlier milestones than AFLW
        gamma=0.1,
        by_epoch=True
    )
]

# Auto-scale learning rate based on batch size
auto_scale_lr = dict(base_batch_size=64)  # Scale based on 64 base batch size

# Evaluators
val_evaluator = dict(
    type='NME',  # Changed to NME like AFLW
    norm_mode='use_norm_item', 
    norm_item='bbox_size'
)
test_evaluator = None

# Hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=1, 
        save_best='NME',  # Changed to NME
        rule='less'       # Lower NME is better
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False)
)

# Runtime settings
random_seed = 42  # Set for reproducibility 