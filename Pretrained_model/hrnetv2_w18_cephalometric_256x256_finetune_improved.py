_base_ = ['./td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py'] # Inherit from original

# Fine-tuning specific
load_from = 'Pretrained_model/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth'
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=2)

# Improved optimizer settings with separate learning rates for classification head
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=3e-4),
    clip_grad=dict(max_norm=5., norm_type=2),
    # Different learning rate for classification head
    paramwise_cfg=dict(
        custom_keys={
            'head.classification_head': dict(lr_mult=10.0),  # 10x learning rate for classification
            'head.feature_adapter': dict(lr_mult=5.0),      # 5x learning rate for adapter
            'head.auxiliary_head': dict(lr_mult=10.0),      # 10x learning rate for auxiliary
        }
    )
)

# Learning rate scheduler with warm-up and step decay
param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, start_factor=1e-3, by_epoch=False),  # Warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=100,
        by_epoch=True,
        milestones=[40, 70, 90],  # More gradual decay
        gamma=0.1)
]

# Dataset settings
dataset_type = 'CustomCephalometricDataset'
data_root = "/content/drive/MyDrive/Lala\'s Masters/"

# Codec - Higher resolution for better precision
codec = dict(
    type='MSRAHeatmap',
    input_size=(384, 384),
    heatmap_size=(96, 96),
    sigma=3)

# Model with Improved Classification Head
model = dict(
    # Ensure we keep the neck from base config
    neck=dict(concat=True, type='FeatureMapProcessor'),
    head=dict(
        type='HRNetV2WithClassificationImproved',  # Use improved head
        in_channels=270,  # HRNet with neck outputs 270 channels
        out_channels=19,  # Number of keypoints
        # Conv layer parameters from base config
        conv_out_channels=(270,),
        conv_kernel_sizes=(1,),
        deconv_out_channels=None,
        # Classification head parameters - IMPROVED
        num_classes=3,  # Skeletal Class I, II, III
        classification_hidden_dim=256,
        classification_dropout=0.3,  # Slightly higher dropout
        classification_loss_weight=2.0,  # Increased weight
        # Class weights for balanced training based on actual dataset distribution
        # From diagnosis: Class I=152, Class II=77, Class III=22 samples
        # Using inverse frequency weighting normalized to sum to 3.0
        class_weights=[0.55, 1.08, 3.79],  # [Class I, Class II, Class III]
        use_feature_adapter=True,  # Use feature adaptation
        auxiliary_loss_weight=0.3,  # Auxiliary loss weight
        # Keypoint detection loss
        loss=dict(
            type='AdaptiveWingLoss',
            alpha=2.1, omega=24., epsilon=1., theta=0.5,
            use_target_weight=False, loss_weight=1.0),
        # Decoder for heatmap
        decoder=codec
    )
)

# Enhanced pipelines with stronger augmentation
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
                   'input_center', 'input_scale', 'input_size', 
                   'patient_text_id', 'set', 'class', 'gt_classification'))
]

val_pipeline = [
    dict(type='LoadImageNumpy'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='CustomPackPoseInputs', 
         meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape', 
                   'bbox', 'bbox_scores', 'flip_indices', 'center', 'scale', 
                   'input_center', 'input_scale', 'input_size', 
                   'patient_text_id', 'set', 'class', 'gt_classification'))
]

test_pipeline = val_pipeline

# DataLoaders with reduced batch size for higher resolution
train_dataloader = dict(
    batch_size=16,  # Reduced for better gradient accumulation
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_mode='topdown',
        pipeline=train_pipeline,
        ann_file='',  # To be populated by training script
    ))

val_dataloader = dict(
    batch_size=16,
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

# Evaluators - Add classification metrics
val_evaluator = [
    dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=[0, 1]),  # Keypoint evaluation
    dict(type='ClassificationMetric', num_classes=3)  # Classification evaluation
]
test_evaluator = val_evaluator

# Hooks for monitoring
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),  # Log every 10 iterations
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,  # Save every 5 epochs
        save_best='NME',  # Save best based on NME
        rule='less'  # Lower NME is better
    )
)

# Custom hooks including concurrent MLP training
custom_hooks = [
    dict(
        type='ConcurrentMLPTrainingHook',
        mlp_epochs=100,
        mlp_batch_size=16,
        mlp_lr=1e-5,
        mlp_weight_decay=1e-4,
        hard_example_threshold=5.0,
        hrnet_hard_example_weight=2.0,
        log_interval=20
    )
]

# Logging configuration to monitor classification performance
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')  # Optional: for TensorBoard visualization
    ]
) 