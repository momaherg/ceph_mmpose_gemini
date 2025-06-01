_base_ = './hrnetv2_w18_cephalometric_256x256_finetune.py' # Inherits 384x384 AdaptiveWingLoss setup

# Experiment 4: 384x384 + CutMix + RandomErasing + AdaptiveWingLoss

# Keep AdaptiveWingLoss from base, sigma=3
# Keep LR=3e-4, batch_size=20, 60 epochs from base

# Augmentations: Add CutMix and RandomErasing
# The base train_pipeline is:
# train_pipeline = [
#     dict(type='LoadImageNumpy'),
#     dict(type='GetBBoxCenterScale'),
#     dict(type='RandomFlip', direction='horizontal'),
#     dict(type='RandomBBoxTransform', shift_prob=0, rotate_factor=30, scale_factor=(0.7, 1.3)),
#     dict(type='TopdownAffine', input_size=(384,384)), # Uses codec['input_size'] which is 384x384
#     dict(type='GenerateTarget', encoder=codec), # Uses codec with sigma=3
#     dict(type='CustomPackPoseInputs', ...)
# ]

# Insert new augmentations before TopdownAffine
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
    # Added Augmentations
    dict(type='CutMix', prob=0.5, alpha=1.0, 
         # Ensure CutMix is compatible with heatmap generation. 
         # It might need specific handling or be applied earlier if issues arise.
    ),
    dict(type='RandomErasing', erase_prob=0.4, 
         scale=(0.02, 0.2), 
         ratio=(0.3, 3.3)
    ),
    dict(type='TopdownAffine', input_size=(384, 384)),
    dict(type='GenerateTarget', encoder={{_base_.codec}}), # Use codec from base (sigma=3)
    dict(type='CustomPackPoseInputs', meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape', 'bbox', 'bbox_scores', 'flip_indices', 'center', 'scale', 'input_center', 'input_scale', 'input_size', 'patient_text_id', 'set', 'class'))
]

# Ensure dataloaders use this modified pipeline
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

# Note: CutMix and RandomErasing are typically not applied during validation/testing.
# The val_pipeline from the base is usually fine. 