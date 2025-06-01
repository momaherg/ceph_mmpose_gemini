_base_ = './hrnetv2_w18_cephalometric_256x256_finetune.py' # Inherits 384x384 setup

# Experiment 1: 384x384 + FocalHeatmapLoss + OHKMMSELoss

# Codec: Sigma back to 2 for FocalHeatmapLoss
codec = dict(
    type='MSRAHeatmap',
    input_size=(384, 384),
    heatmap_size=(96, 96),
    sigma=2  # FocalHeatmapLoss prefers sharper targets
)

# Model: Composite Loss
model = dict(
    head=dict(
        loss=dict(
            type='CombinedLoss',
            losses=[
                dict(type='FocalHeatmapLoss', alpha=2, beta=4, loss_weight=1.0),
                dict(type='KeypointOHKMMSELoss', ohkm_ratio=0.25, use_target_weight=True, loss_weight=1.0)
            ]
        )
    )
)

# Optimizer: Adjusted LR
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1.5e-4), # Adjusted from 3e-4
    clip_grad=dict(max_norm=5., norm_type=2)
)

# Training schedule: 80 epochs
train_cfg = dict(by_epoch=True, max_epochs=80, val_interval=2)

# LR scheduler: Update T_max for new epoch count
param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, start_factor=1e-3, by_epoch=False),
    dict(type='CosineAnnealingLR', T_max=80, eta_min=1e-6, by_epoch=True) # T_max matches max_epochs
]

# Pipelines need to use the updated codec for GenerateTarget
# The _base_ config already has input_size=(384,384) in TopdownAffine, which is fine.
# We only need to ensure GenerateTarget uses the new sigma.
train_pipeline_modifier = dict(
    type='TransformBroadcaster',
    transforms=[
        dict(type='GenerateTarget', encoder=codec) # Override GenerateTarget to use new sigma
    ]
)

# Find the GenerateTarget step and replace it or update its encoder
# This is a bit complex to do with simple dict updates if GenerateTarget is deep in the list.
# Assuming GenerateTarget is the second to last transform as in the base config.
# A more robust way is to rebuild the pipeline, but let's try modifying for now.

# For simplicity, we rely on the fact that `train_pipeline` is defined in the _base_ and then modified here.
# If GenerateTarget is not the second to last, this will need adjustment.
# The most straightforward way if _base_ train_pipeline is complex is to redefine it fully.

# Re-define train_pipeline to ensure codec with sigma=2 is used by GenerateTarget
# (Copying from base and modifying GenerateTarget's encoder)
train_pipeline = [
    dict(type='LoadImageNumpy'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(
        type='RandomBBoxTransform',
        shift_prob=0,
        rotate_factor=30,
        scale_factor=(0.7, 1.3)),
    dict(type='TopdownAffine', input_size=(384, 384)), # from base or explicit here
    dict(type='GenerateTarget', encoder=codec), # <<< This uses the new codec with sigma=2
    dict(type='CustomPackPoseInputs', meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape', 'bbox', 'bbox_scores', 'flip_indices', 'center', 'scale', 'input_center', 'input_scale', 'input_size', 'patient_text_id', 'set', 'class'))
]

# Val pipeline also needs GenerateTarget for some evals if not in test_mode, but usually not.
# If val pipeline includes GenerateTarget, it should also be updated.
# For now, assume val_pipeline from base is okay as it usually doesn't generate targets. 