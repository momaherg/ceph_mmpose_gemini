_base_ = ['../../Pretrained_model/td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py']

# Stage 1: Train at 256x256 resolution
load_from = 'Pretrained_model/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth'
train_cfg = dict(by_epoch=True, max_epochs=40, val_interval=2)  # 40 epochs for stage 1

# Optimizer settings
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=3e-4),  # Higher LR for initial training
    clip_grad=dict(max_norm=5., norm_type=2)
)

# Learning rate scheduler  
param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, start_factor=1e-3, by_epoch=False),
    dict(type='CosineAnnealingLR', T_max=40, eta_min=1e-5, by_epoch=True)
]

# Dataset settings
dataset_type = 'CustomCephalometricDataset'
data_root = "/content/drive/MyDrive/Lala\'s Masters/"

# Codec - Stage 1: 256x256 resolution
codec = dict(
    type='MSRAHeatmap',
    input_size=(256, 256),  # Lower resolution for stage 1
    heatmap_size=(64, 64),
    sigma=2)

# Model with OHKMMSELoss for stage 1
model = dict(
    head=dict(
        out_channels=19,
        loss=dict(
            type='OHKMMSELoss',
            use_target_weight=True,
            topk=8  # Focus on harder examples
        )
    )
)

# Standard pipelines for 256x256
train_pipeline = [
    dict(type='LoadImageNumpy'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(
        type='RandomBBoxTransform',
        shift_prob=0,
        rotate_factor=25,  # Moderate augmentation for stage 1
        scale_factor=(0.75, 1.25)),
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

# DataLoaders - Higher batch size for 256x256
train_dataloader = dict(
    batch_size=32,  # Higher batch size for 256x256
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
    batch_size=32,
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