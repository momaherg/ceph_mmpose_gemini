"""
Experiment A Configuration: AdaptiveWingLoss + OHKM Hybrid at 384x384
This combines the robustness of AdaptiveWingLoss with Online Hard Keypoint Mining
to focus on difficult landmarks like Sella and Gonion.
"""

_base_ = ['../Pretrained_model/td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py']

# Fine-tuning from the best V4 checkpoint
load_from = 'work_dirs/hrnetv2_w18_cephalometric_384x384_adaptive_wing_loss_v4/best_NME_epoch_52.pth'

# Training configuration
train_cfg = dict(by_epoch=True, max_epochs=60, val_interval=2)

# Optimizer with gradient clipping for stability
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-4),  # Conservative LR for hybrid loss
    clip_grad=dict(max_norm=5., norm_type=2)
)

# Learning rate schedule with warm-up and cosine annealing
param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, start_factor=1e-3, by_epoch=False),
    dict(type='CosineAnnealingLR', T_max=60, eta_min=1e-6, by_epoch=True)
]

# Dataset settings
dataset_type = 'CustomCephalometricDataset'
data_root = ""

# High-resolution codec for better precision
codec = dict(
    type='MSRAHeatmap',
    input_size=(384, 384),
    heatmap_size=(96, 96),
    sigma=3
)

# Model with AdaptiveWing+OHKM Hybrid Loss
model = dict(
    head=dict(
        out_channels=19,
        loss=dict(
            type='AdaptiveWingOHKMHybridLoss',
            # Hard keypoint mining parameters
            topk=8,  # Mine top-8 hardest keypoints per sample
            ohkm_weight=2.0,  # 2x weight for hard keypoints
            # AdaptiveWingLoss parameters (tuned for stability)
            alpha=2.1,
            omega=24.0,  # ~1.5× heatmap sigma*8
            epsilon=1.0,
            theta=0.5,
            use_target_weight=True,
            loss_weight=1.0
        )
    )
)

# Enhanced data pipelines with augmentation
train_pipeline = [
    dict(type='LoadImageNumpy'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(
        type='RandomBBoxTransform',
        shift_prob=0,
        rotate_factor=30,
        scale_factor=(0.7, 1.3)
    ),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='CustomPackPoseInputs', 
         meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape', 
                   'bbox', 'bbox_scores', 'flip_indices', 'center', 'scale', 
                   'input_center', 'input_scale', 'input_size', 
                   'patient_text_id', 'set', 'class'))
]

val_pipeline = [
    dict(type='LoadImageNumpy'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='CustomPackPoseInputs', 
         meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape', 
                   'bbox', 'bbox_scores', 'flip_indices', 'center', 'scale', 
                   'input_center', 'input_scale', 'input_size', 
                   'patient_text_id', 'set', 'class'))
]

test_pipeline = val_pipeline

# DataLoaders with reduced batch size for 384x384
train_dataloader = dict(
    batch_size=20,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_mode='topdown',
        pipeline=train_pipeline,
        ann_file=''  # To be populated by training script
    )
)

val_dataloader = dict(
    batch_size=20,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_mode='topdown',
        pipeline=val_pipeline,
        ann_file='',
        test_mode=True
    )
)

test_dataloader = val_dataloader

# Evaluators
val_evaluator = dict(
    type='NME', 
    norm_mode='keypoint_distance', 
    keypoint_indices=[0, 1]  # Sella and Nasion for normalization
)
test_evaluator = val_evaluator

# Experiment tracking
work_dir = 'work_dirs/experiment_a_adaptive_wing_ohkm_hybrid'
experiment_name = 'experiment_a_adaptive_wing_ohkm_hybrid' 