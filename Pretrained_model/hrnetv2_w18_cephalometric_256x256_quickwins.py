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

# QUICK WIN 2: UDP Codec with improved heatmap refinement
codec = dict(
    type='UDPHeatmap',  # UDP (Unbiased Data Processing) for better coordinate accuracy
    input_size=(256, 256), # Crucial: Model expects 256x256 input
    heatmap_size=(64, 64),
    sigma=2,
    use_udp=True,  # Enable unbiased data processing
    target_type='GaussianHeatmap'
)

# Model head output channels should match number of keypoints (19 for your dataset)
model = dict(
    head=dict(
        out_channels=19, # Ensure this matches your dataset's keypoint count
        loss=dict(type='KeypointMSELoss', use_target_weight=True)
    ),
    # QUICK WIN 3: Enhanced test config for test-time augmentation
    test_cfg=dict(
        flip_test=True,  # Enable horizontal flip test-time augmentation
        flip_mode='heatmap',  # Use heatmap-based flipping
        shift_heatmap=True  # Enable sub-pixel shifting for better accuracy
    )
)

# Enhanced pipelines with stronger augmentation
train_pipeline = [
    dict(type='LoadImageNumpy'), # Load from numpy array
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(
        type='RandomBBoxTransform',
        shift_prob=0,
        rotate_factor=30, # Increased rotation for better generalization
        scale_factor=(0.7, 1.3)), # Wider scale range
    dict(type='TopdownAffine', input_size=codec['input_size']), # Use 256x256
    dict(type='GenerateTarget', encoder=codec),
    dict(type='CustomPackPoseInputs', meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape', 'bbox', 'bbox_scores', 'flip_indices', 'center', 'scale', 'input_center', 'input_scale', 'input_size', 'patient_text_id', 'set', 'class'))
]

val_pipeline = [
    dict(type='LoadImageNumpy'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']), # Use 256x256
    dict(type='CustomPackPoseInputs', meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape', 'bbox', 'bbox_scores', 'flip_indices', 'center', 'scale', 'input_center', 'input_scale', 'input_size', 'patient_text_id', 'set', 'class'))
]

# Test pipeline with additional augmentation support
test_pipeline = [
    dict(type='LoadImageNumpy'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='CustomPackPoseInputs', meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape', 'bbox', 'bbox_scores', 'flip_indices', 'center', 'scale', 'input_center', 'input_scale', 'input_size', 'patient_text_id', 'set', 'class'))
]

# DataLoaders
train_dataloader = dict(
    batch_size=32, # Adjust as per your GPU memory, original was 64
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_mode='topdown',
        pipeline=train_pipeline,
        ann_file='', # To be populated by filtered data_df or if dataset handles split from one file
    ))

val_dataloader = dict(
    batch_size=32, # Adjust as per your GPU memory
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_mode='topdown',
        pipeline=val_pipeline,
        ann_file='',
        test_mode=True # Crucial for validation
    ))

test_dataloader = dict(
    batch_size=16, # Smaller batch for test-time augmentation
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_mode='topdown',
        pipeline=test_pipeline,
        ann_file='',
        test_mode=True
    ))

# Evaluators
val_evaluator = dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=[0, 1]) # Use Sella and Nasion for normalization
test_evaluator = val_evaluator

# Visualization (optional, but good for debugging)
# visualizer = dict(
#    type='PoseLocalVisualizer',
#    vis_backends=[dict(type='LocalVisBackend')],
#    name='visualizer')
# default_hooks = dict(
#    visualization=dict(enable=True, type='PoseVisualizationHook')) 