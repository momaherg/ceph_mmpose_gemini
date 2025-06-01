_base_ = ['./td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py'] # Inherit from original

# Fine-tuning specific
load_from = 'Pretrained_model/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth'
train_cfg = dict(by_epoch=True, max_epochs=60, val_interval=2) # Longer training with less frequent validation

# Improved optimizer settings with cosine annealing
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=3e-4), # Higher starting LR for cosine schedule
    # clip_grad=None
)

# Learning rate scheduler with warm-up and cosine annealing  
param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, start_factor=1e-3, by_epoch=False),  # Warm-up
    dict(type='CosineAnnealingLR', T_max=60, eta_min=1e-6, by_epoch=True)  # Cosine annealing
]

# Dataset settings
dataset_type = 'CustomCephalometricDataset' # Your custom dataset
data_root = "/content/drive/MyDrive/Lala\'s Masters/" # Conventional data root, actual data comes from data_df injected by training script

# Codec - ORIGINAL RESOLUTION 256x256 (not upgraded)
codec = dict(
    type='MSRAHeatmap',
    input_size=(256, 256), # ORIGINAL: 256x256 resolution
    heatmap_size=(64, 64),  # ORIGINAL: 64x64 heatmaps
    sigma=2)

# Model head with KeypointMSELoss (simple MSE for comparison)
model = dict(
    head=dict(
        out_channels=19, # Ensure this matches your dataset's keypoint count
        loss=dict(
            type='KeypointMSELoss', # Standard MSE loss for baseline
            use_target_weight=True, # Works with joint_weights for Sella/Gonion emphasis
        )
    )
    # The rest of the model (backbone, neck, data_preprocessor, test_cfg)
    # can be inherited or slightly adjusted if needed.
)

# Enhanced pipelines with stronger augmentation but original resolution
train_pipeline = [
    dict(type='LoadImageNumpy'), # Load from numpy array
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(
        type='RandomBBoxTransform',
        shift_prob=0,
        rotate_factor=30, # Increased rotation for better generalization
        scale_factor=(0.7, 1.3)), # Wider scale range
    dict(type='TopdownAffine', input_size=codec['input_size']), # Uses 256x256
    dict(type='GenerateTarget', encoder=codec),
    dict(type='CustomPackPoseInputs', meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape', 'bbox', 'bbox_scores', 'flip_indices', 'center', 'scale', 'input_center', 'input_scale', 'input_size', 'patient_text_id', 'set', 'class'))
]
val_pipeline = [
    dict(type='LoadImageNumpy'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']), # Uses 256x256
    dict(type='CustomPackPoseInputs', meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape', 'bbox', 'bbox_scores', 'flip_indices', 'center', 'scale', 'input_center', 'input_scale', 'input_size', 'patient_text_id', 'set', 'class'))
]
test_pipeline = val_pipeline # Test pipeline often same as validation

# DataLoaders - ORIGINAL batch size 32 (since we're using 256x256)
train_dataloader = dict(
    batch_size=32, # ORIGINAL: 32 batch size for 256x256
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='', # To be populated by filtered data_df
        data_mode='topdown',
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=32, # ORIGINAL: 32 batch size for 256x256
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file='',
        data_mode='topdown',
        pipeline=val_pipeline,
        test_mode=True # Crucial for validation
    ))
test_dataloader = val_dataloader

# Evaluators
val_evaluator = dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=[0, 1]) # Use Sella and Nasion for normalization
test_evaluator = val_evaluator 