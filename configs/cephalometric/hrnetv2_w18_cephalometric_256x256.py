# Inherit from the default runtime and the AFLW model config for structure
_base_ = [
    '../_base_/default_runtime.py',
]

# Paths and dataset types
# Make sure 'cephalometric_dataset.py' is in your PYTHONPATH or a discoverable location by MMPose
# For example, if it's in the same directory as your main script, you might need to add that path.
# Or place it in mmpose/datasets/
custom_imports = dict(imports=['cephalometric_dataset'], allow_failed_imports=False)

dataset_type = 'CephalometricDataset'
metainfo_file = 'mmpose_ceph_data/cephalometric_dataset_info.py' # Relative to workspace root
data_root = 'mmpose_ceph_data/' # Relative to workspace root

# Pre-trained model URL
pretrained_model_path = 'https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth'

# Model settings from AFLW config, adapted for fine-tuning
channel_cfg = dict(
    num_output_channels=19, # Matches your dataset
    dataset_joints=19,      # Matches your dataset
    dataset_channel=[
        list(range(19)),
    ],
    inference_channel=list(range(19)))

model = dict(
    type='TopDown',
    # No 'pretrained' backbone here, as init_cfg will load the full model checkpoint
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
            upsample=dict(mode='bilinear', align_corners=False))),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=[18, 36, 72, 144],
        in_index=(0, 1, 2, 3),
        input_transform='resize_concat',
        out_channels=channel_cfg['num_output_channels'], # Should be 19
        num_deconv_layers=0,
        extra=dict(
            final_conv_kernel=1, num_conv_layers=1, num_conv_kernels=(1, )),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11),
    init_cfg=dict(type='Pretrained', checkpoint=pretrained_model_path) # Load the full pre-trained model
)

# Data configuration from AFLW, adapted
data_cfg = dict(
    image_size=[256, 256],       # Resize 224x224 to 256x256 to match pre-training
    heatmap_size=[64, 64],       # Standard heatmap size for this model
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False, # From AFLW
    nms_thr=0.9,    # From AFLW
    oks_thr=0.9,    # From AFLW
    vis_thr=0.2,    # From AFLW
    use_gt_bbox=True, # Use GT bboxes from our annotations
    bbox_file=None,   # Not needed as we use GT bboxes
    space_color=[255, 0, 0], # From AFLW (visualization color)
    mean=[0.485, 0.456, 0.406], # ImageNet mean
    std=[0.229, 0.224, 0.225],  # ImageNet std
    # flip_pairs: This will be loaded from metainfo by the dataset class
)

# Training pipeline (adapted from AFLW)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # Our bboxes are already quite good (derived from keypoints or full image).
    # TopDownGetBboxCenterScale can still be useful for consistent padding/aspect ratio.
    # Padding can be adjusted. AFLW used 1.25. Let's start with a smaller padding if bboxes are tight.
    dict(type='TopDownGetBboxCenterScale', padding=1.1),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    # Augmentations: adjust factors as needed for cephalometric images
    dict(type='TopDownGetRandomScaleRotation', rot_factor=30, scale_factor=0.25),
    dict(type='TopDownAffine', use_udp=True, input_size=data_cfg['image_size']),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=data_cfg['mean'], std=data_cfg['std']),
    dict(type='TopDownGenerateTarget', sigma=2, use_different_joint_weights=False), # sigma=2 is common for 256x256 input -> 64x64 heatmap
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'img_path', 'img_id', 'joints_3d', 'joints_3d_visible', 'center',
            'scale', 'rotation', 'bbox', 'flip_pairs'
        ]),
]

# Validation pipeline (adapted from AFLW)
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.1),
    dict(type='TopDownAffine', use_udp=True, input_size=data_cfg['image_size']),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=data_cfg['mean'], std=data_cfg['std']),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'img_path', 'img_id', 'center', 'scale', 'rotation', 'bbox', 'flip_pairs'
        ]),
]

# Dataloaders
train_dataloader = dict(
    batch_size=32, # Adjust based on GPU memory
    num_workers=4,  # Adjust based on CPU cores
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/train_annotations.json',
        metainfo_file=metainfo_file, # Added this
        data_root=data_root,
        data_prefix=dict(img=''), # img_path in annotations is relative to data_root
        data_cfg=data_cfg,
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/val_annotations.json',
        metainfo_file=metainfo_file, # Added this
        data_root=data_root,
        data_prefix=dict(img=''),
        data_cfg=data_cfg,
        pipeline=val_pipeline))

test_dataloader = val_dataloader # Use val_dataloader for testing as well

# Evaluation metrics
# NME (Normalized Mean Error) is common. PCK (Percentage of Correct Keypoints) and AUC (Area Under Curve) are also useful.
# Ensure your `dataset_info` has `sigmas` defined if you use OKS-based NME or PCK variants.
# The `CephalometricDataset` will load `sigmas` from `metainfo_file`.
# The evaluator will use these if `use_area_keypoint_error=True` for NME, or for OKS calculation in general.
evaluation = dict(interval=1, metric=['NME', 'PCK', 'AUC'], save_best='NME')

# Optimizer (from default_runtime, can be overridden here if needed)
# optimizer = dict(type='Adam', lr=2e-3) # Example: AFLW used 2e-3

# Learning rate schedule (from default_runtime, can be overridden)
# lr_config = dict(policy='step', warmup='linear', warmup_iters=500, warmup_ratio=0.001, step=[40, 55])
# total_epochs = 60 # Example: AFLW used 60

# Checkpoint config (from default_runtime)
# checkpoint_config = dict(interval=1)

# Log config (from default_runtime)
# log_config = dict(
#     interval=5, # Log every 5 iterations for AFLW
#     hooks=[
#         dict(type='TextLoggerHook'),
#         # dict(type='TensorboardLoggerHook')
#     ]) 