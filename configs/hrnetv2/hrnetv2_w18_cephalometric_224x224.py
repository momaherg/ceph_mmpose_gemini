# Inherit from default runtime settings
_base_ = ['../_base_/default_runtime.py']

# Import custom dataset meta-information
# This assumes cephalometric_dataset_info.py is in Python's import path
# (e.g., in the same directory as main.py or an installed package)
from cephalometric_dataset_info import dataset_info as cephalometric_metainfo

# Model settings
# Pretrained model path
pretrained_model = 'https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth'

# Input image size
input_size = (224, 224)

# Number of keypoints
num_keypoints = 19 # From your dataset_info

# Model definition
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53], # Default MMEngine/MMCV mean
        std=[58.395, 57.12, 57.375],   # Default MMEngine/MMCV std
        bgr_to_rgb=True # if your LoadImageNumpy provides RGB, and preprocessor expects BGR input for mean/std
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
        type='HeatmapHead', # MMPose 1.x HeatmapHead
        in_channels=[18, 36, 72, 144], # Corresponds to HRNet stage4 output
        input_transform='resize_concat', # How to process multi-scale features
        out_channels=num_keypoints,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=dict( # Decoder for converting heatmaps to coordinates
            type='MegviiHeatmap', # Or IntegralRegression, SimCCDecoder, etc.
            input_size=input_size,
            heatmap_size=(input_size[0] // 4, input_size[1] // 4), # e.g., 56x56 for 224x224 input
            kernel_size=7, # For UDP or heatmap processing, adjust if needed
            # sigma=2, # Sigma for Gaussian target generation is in pipeline, this is for decoding
        )
    ),
    test_cfg=dict(
        flip_test=False,  # Disabled - cephalometric landmarks are not symmetric
        flip_mode='heatmap',
        shift_heatmap=True, # From AFLW config
    )
)

# Dataset settings
dataset_type = 'CustomCephalometricDataset'
data_root = '/content/drive/MyDrive/Lala\'s Masters/' # Base path for your data JSON files

# Pipelines
# Note: `LoadImageNumpy` is a custom transform we defined.
# `GenerateTarget` creates heatmaps for training.
# `PackPoseInputs` prepares data for the model.

common_pipeline_prefix = [
    dict(type='LoadImageNumpy'),  # Our custom transform - will attempt to use this first
    # If LoadImageNumpy fails, provide a fallback option with mmpose's Identity transform
    # dict(type='Identity'),  # Uncomment this if LoadImageNumpy fails to be registered
    # Assuming CustomCephalometricDataset provides 'bbox' as [0, 0, 224, 224] or similar
    dict(type='GetBBoxCenterScale'), # Changed from TopDownGetBboxCenterScale
]

common_pipeline_suffix = [
    dict(type='TopdownAffine', input_size=input_size),
    dict(type='GenerateTarget', 
         encoder=dict(
            type='MSRAHeatmap',
            input_size=input_size, 
            heatmap_size=(input_size[0] // 4, input_size[1] // 4),
            sigma=2 # Sigma for Gaussian heatmap generation
        )
    ),
    dict(type='CustomPackPoseInputs', 
         meta_keys=(
             'img_id', 'img_path', 'ori_shape', 'img_shape',
             'input_size', 'input_center', 'input_scale',
             'flip', 'flip_direction',
             'num_joints', 'joint_weights', # from metainfo
             'id', 'patient_text_id', 'set', 'class' # Custom keys from dataset
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

train_pipeline = [
    *common_pipeline_prefix,
    dict(type='RandomBBoxTransform', # Includes scaling and rotation
         scale_factor=[0.8, 1.2],
         rotate_factor=15,
         shift_factor=0.0),
    *common_pipeline_suffix
]

val_pipeline = [
    *common_pipeline_prefix,
    *val_test_pipeline_suffix
]

test_pipeline = val_pipeline

# Dataloaders
train_dataloader = dict(
    batch_size=32, # Adjust to your GPU memory
    num_workers=4,  # Adjust to your CPU cores
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_data_pure_old_numpy.json', # Relative to data_root
        metainfo=cephalometric_metainfo, # Pass the imported metainfo
        pipeline=train_pipeline,
        test_mode=False,
    )
)

val_dataloader = dict(
    batch_size=16,
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
    thr=0.05,  # 5% threshold for PCK accuracy
    norm_item=['bbox', 'torso']
)
test_evaluator = None

# Training schedule
train_cfg = dict(by_epoch=True, max_epochs=60, val_interval=5)
val_cfg = dict()  # Enable validation
test_cfg = None

# Optimizer
optim_wrapper = dict(optimizer=dict(
    type='AdamW',
    lr=1e-4, # Reduced from 2e-4 - critical fix for model performance
    weight_decay=0.0001
))

# Learning rate scheduler - Changed to gentler CosineAnnealingLR
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500 # Warmup iterations
    ),
    dict(
        type='CosineAnnealingLR',  # Changed from aggressive MultiStepLR
        begin=0, 
        end=60, # Total epochs
        by_epoch=True,
        T_max=60,
        eta_min=1e-6  # Minimum learning rate
    )
]

# Hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='PCKAccuracy', rule='greater'), # Save best based on PCK
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False) # Set enable=True to visualize predictions
)

# Custom hooks if any, e.g. for Tensorboard
# custom_hooks = [dict(type='TensorboardLoggerHook', ndarray_as_scalar=False)]

# Runtime settings
# random_seed = 0 # For reproducibility
# find_unused_parameters = True # If DDP complains 