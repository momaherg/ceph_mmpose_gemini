_base_ = ['../../Pretrained_model/td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py']

# Fine-tuning specific
load_from = 'Pretrained_model/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth'
train_cfg = dict(by_epoch=True, max_epochs=60, val_interval=2)

# Optimizer settings with gradient clipping
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=2e-4),  # Moderate LR for combined loss
    clip_grad=dict(max_norm=5., norm_type=2)
)

# Learning rate scheduler with warm-up and cosine annealing  
param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, start_factor=1e-3, by_epoch=False),
    dict(type='CosineAnnealingLR', T_max=60, eta_min=1e-6, by_epoch=True)
]

# Dataset settings
dataset_type = 'CustomCephalometricDataset'
data_root = "/content/drive/MyDrive/Lala\'s Masters/"

# Codec - High resolution for precise localization
codec = dict(
    type='MSRAHeatmap',
    input_size=(384, 384),
    heatmap_size=(96, 96),
    sigma=2.5)  # Slightly smaller sigma for sharper peaks with FocalLoss

# Model with Combined FocalHeatmapLoss + OHKM Loss
model = dict(
    head=dict(
        out_channels=19,
        loss=[
            dict(
                type='FocalHeatmapLoss',
                alpha=2,
                gamma=4,
                use_target_weight=True,
                loss_weight=0.7  # Primary loss
            ),
            dict(
                type='OHKMMSELoss',
                use_target_weight=True,
                topk=5,  # Focus on top 5 hardest keypoints
                loss_weight=0.3  # Supplementary loss for hard examples
            )
        ]
    )
)

# Enhanced pipelines
train_pipeline = [
    dict(type='LoadImageNumpy'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(
        type='RandomBBoxTransform',
        shift_prob=0,
        rotate_factor=30,
        scale_factor=(0.7, 1.3)),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='CustomPackPoseInputs', 
         meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape', 
                   'bbox', 'bbox_scores', 'flip_indices', 'center', 'scale', 
                   'input_center', 'input_scale', 'input_size', 'patient_text_id', 
                   'set', 'class'))
]

val_pipeline = [
    dict(type='LoadImageNumpy'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='CustomPackPoseInputs',
         meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape',
                   'bbox', 'bbox_scores', 'flip_indices', 'center', 'scale',
                   'input_center', 'input_scale', 'input_size', 'patient_text_id',
                   'set', 'class'))
]

test_pipeline = val_pipeline

# DataLoaders
train_dataloader = dict(
    batch_size=20,  # For 384x384 resolution
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_mode='topdown',
        pipeline=train_pipeline,
        ann_file='',
    ))

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
    ))

test_dataloader = val_dataloader

# Evaluators
val_evaluator = dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=[0, 1])
test_evaluator = val_evaluator 