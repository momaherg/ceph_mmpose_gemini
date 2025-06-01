# This config is based on your *original* 256x256 setup with OHKM if you had one,
# or a simplified 256x256 version of your current best config.
# For simplicity, I will assume a base 256x256 config and apply OHKM.

_base_ = './hrnetv2_w18_cephalometric_256x256_finetune.py' # Base config with 256x256 settings

# Modify the base to be 256x256 if it isn't already, and use OHKM
codec = dict(
    type='MSRAHeatmap',
    input_size=(256, 256),
    heatmap_size=(64, 64),
    sigma=2 # Typical sigma for 256x256
)

model = dict(
    head=dict(
        loss=dict(
            type='KeypointOHKMMSELoss',
            ohkm_ratio=0.25,
            use_target_weight=True,
            loss_weight=1.0
        )
    )
)

# Pipelines for 256x256
train_pipeline = [
    dict(type='LoadImageNumpy'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(
        type='RandomBBoxTransform',
        shift_prob=0,
        rotate_factor=30,
        scale_factor=(0.7, 1.3)),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='CustomPackPoseInputs', meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape', 'bbox', 'bbox_scores', 'flip_indices', 'center', 'scale', 'input_center', 'input_scale', 'input_size', 'patient_text_id', 'set', 'class'))
]
val_pipeline = [
    dict(type='LoadImageNumpy'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(256, 256)),
    dict(type='CustomPackPoseInputs', meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape', 'bbox', 'bbox_scores', 'flip_indices', 'center', 'scale', 'input_center', 'input_scale', 'input_size', 'patient_text_id', 'set', 'class'))
]
test_pipeline = val_pipeline

# Dataloaders batch size for 256x256 (can be higher)
train_dataloader = dict(batch_size=32, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=32, dataset=dict(pipeline=val_pipeline))
test_dataloader = dict(batch_size=32, dataset=dict(pipeline=test_pipeline))

# Training: 10 epochs
train_cfg = dict(by_epoch=True, max_epochs=10, val_interval=1)

# Optimizer: LR for 256x256 phase
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-4),
    clip_grad=dict(max_norm=5., norm_type=2)
)

# LR Scheduler: Simple for 10 epochs
param_scheduler = [
    dict(type='LinearLR', begin=0, end=100, start_factor=1e-3, by_epoch=False),
    dict(type='CosineAnnealingLR', T_max=10, eta_min=1e-6, by_epoch=True)
]

# Crucially, ensure load_from points to the initial ImageNet/AFLW pretrain, not a custom finetune yet.
load_from = 'Pretrained_model/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth' 