# This config starts from the checkpoint of exp2_stage1_256_ohkm.py
# The `load_from` path will need to be updated by the master script.
_base_ = './hrnetv2_w18_cephalometric_256x256_finetune.py' # Base config with 384x384 and AdaptiveWingLoss

# Experiment 2, Stage 2: 384x384, 50 epochs, AdaptiveWingLoss, load from Stage 1

# `load_from` will be set by the master script to the best checkpoint from Stage 1
# load_from = 'work_dirs/exp2_stage1_256_ohkm/best_NME_epoch_10.pth' # Example path

# Codec: 384x384, sigma=3 (as in the base 384x384 config)
codec = dict(
    type='MSRAHeatmap',
    input_size=(384, 384),
    heatmap_size=(96, 96),
    sigma=3
)

# Model: AdaptiveWingLoss (as in the base 384x384 config)
model = dict(
    head=dict(
        loss=dict(
            type='AdaptiveWingLoss',
            alpha=2.1,
            omega=24.0,
            epsilon=1.0,
            theta=0.5,
            use_target_weight=False # As per your previous successful 384 config
        )
    )
)

# Pipelines for 384x384
train_pipeline = [
    dict(type='LoadImageNumpy'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(
        type='RandomBBoxTransform',
        shift_prob=0,
        rotate_factor=30,
        scale_factor=(0.7, 1.3)),
    dict(type='TopdownAffine', input_size=(384, 384)),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='CustomPackPoseInputs', meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape', 'bbox', 'bbox_scores', 'flip_indices', 'center', 'scale', 'input_center', 'input_scale', 'input_size', 'patient_text_id', 'set', 'class'))
]
val_pipeline = [
    dict(type='LoadImageNumpy'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(384, 384)),
    dict(type='CustomPackPoseInputs', meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape', 'bbox', 'bbox_scores', 'flip_indices', 'center', 'scale', 'input_center', 'input_scale', 'input_size', 'patient_text_id', 'set', 'class'))
]
test_pipeline = val_pipeline

# Dataloaders batch size for 384x384
train_dataloader = dict(batch_size=20, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=20, dataset=dict(pipeline=val_pipeline))
test_dataloader = dict(batch_size=20, dataset=dict(pipeline=test_pipeline))


# Training: 50 epochs for this stage
train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=2)

# Optimizer: Reset LR for 384x384 fine-tuning phase
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=2e-4),
    clip_grad=dict(max_norm=5., norm_type=2)
)

# LR Scheduler: 50 epochs
param_scheduler = [
    dict(type='LinearLR', begin=0, end=250, start_factor=1e-3, by_epoch=False), # shorter warmup for finetuning
    dict(type='CosineAnnealingLR', T_max=50, eta_min=1e-6, by_epoch=True)
] 