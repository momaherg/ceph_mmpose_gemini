_base_ = ['./td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py'] # Inherit from original

# Custom imports for multi-task model
custom_imports = dict(
    imports=['multitask_cephalometric_model', 'classification_metric'],
    allow_failed_imports=False
)

# Fine-tuning specific
load_from = 'Pretrained_model/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth'
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=2) # Extended training: 100 epochs

# Improved optimizer settings with cosine annealing
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=3e-4), # Higher starting LR for cosine schedule
    clip_grad=dict(max_norm=5.,  # see next section
                   norm_type=2)
)

# Learning rate scheduler with warm-up and step decay
param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, start_factor=1e-3, by_epoch=False),  # Warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=100,
        by_epoch=True,
        milestones=[155, 200],  # Decay LR at epoch 155 and 200 (scaled from [70, 90] for 100 epochs)
        gamma=0.1)
]

# Dataset settings
dataset_type = 'CustomCephalometricDataset' # Your custom dataset
data_root = "/content/drive/MyDrive/Lala\'s Masters/" # Conventional data root, actual data comes from data_df injected by training script

# Codec - UPGRADED: Higher resolution for better precision
codec = dict(
    type='MSRAHeatmap',
    input_size=(384, 384), # UPGRADED: Was (256, 256) - Higher resolution for finer details
    heatmap_size=(96, 96),  # UPGRADED: Was (64, 64) - Larger heatmaps for sub-pixel precision
    sigma=3)

# Model configuration - Multi-task model with classification
model = dict(
    type='MultiTaskCephalometricModel',  # Our custom multi-task model
    classification_loss_weight=1.0,  # Weight for classification loss
    use_landmark_features_for_classification=True,  # Use both backbone features and landmarks
    # Override head configuration for multi-task
    head=dict(
        out_channels=19, # Ensure this matches your dataset's keypoint count
        loss=dict(  # main loss
                type='AdaptiveWingLoss',
                alpha=2.1,  omega=24., epsilon=1., theta=0.5,
                use_target_weight=False, loss_weight=1.0)
    ),
    # Add classification head configuration
    classification_head=dict(
        type='ClassificationHead',
        in_channels=270,  # HRNet-W18 concatenated features: 18+36+72+144
        num_classes=3,    # Class I, II, III
        hidden_dim=256,
        dropout_rate=0.2
    )
    # The rest of the model (backbone, neck, data_preprocessor, test_cfg)
    # can be inherited or slightly adjusted if needed.
)

# Enhanced pipelines with stronger augmentation and higher resolution
train_pipeline = [
    dict(type='LoadImageNumpy'), # Load from numpy array
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(
        type='RandomBBoxTransform',
        shift_prob=0,
        rotate_factor=30, # Increased rotation for better generalization
        scale_factor=(0.7, 1.3)), # Wider scale range
    dict(type='TopdownAffine', input_size=codec['input_size']), # Now uses 384x384
    dict(type='GenerateTarget', encoder=codec),
    dict(type='CustomPackPoseInputs', meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape', 'bbox', 'bbox_scores', 'flip_indices', 'center', 'scale', 'input_center', 'input_scale', 'input_size', 'patient_text_id', 'set', 'class'))
]
val_pipeline = [
    dict(type='LoadImageNumpy'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']), # Now uses 384x384
    dict(type='CustomPackPoseInputs', meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape', 'bbox', 'bbox_scores', 'flip_indices', 'center', 'scale', 'input_center', 'input_scale', 'input_size', 'patient_text_id', 'set', 'class'))
]
test_pipeline = val_pipeline # Test pipeline often same as validation

# DataLoaders - REDUCED batch size for higher resolution
train_dataloader = dict(
    batch_size=10, # REDUCED: Was 20 - Lowered to prevent OOM with multi-task model
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_mode='topdown',
        pipeline=train_pipeline,
        ann_file='', # To be populated by filtered data_df 
        test_mode=False
    ))

val_dataloader = dict(
    batch_size=10, # REDUCED: Was 20 - Matching training batch size
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
test_dataloader = val_dataloader # Often, test and val dataloaders are configured similarly

# Evaluators - Add classification metrics
val_evaluator = [
    dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=[0, 1]),  # Use Sella and Nasion for normalization
    dict(
        type='ClassificationMetric',
        num_classes=3,
        mode='macro',
        compute_per_class=True
    )
]
test_evaluator = val_evaluator

# =========================================================================
# CONCURRENT JOINT MLP TRAINING HOOK WITH CLASSIFICATION
# =========================================================================
# This hook trains a joint MLP refinement model on-the-fly during HRNetV2 training.
# The MLP now also predicts patient classification (Class I, II, III).
# After each HRNet epoch, it:
# 1. Runs inference on training data using current HRNet weights
# 2. Identifies hard examples based on landmark prediction errors
# 3. Trains a joint 38-D MLP model for 100 epochs with both regression and classification losses
# 4. Creates weighted sampler for next HRNet epoch to oversample hard examples
# 5. Keeps MLP parameters independent (no gradient leakage to HRNet)

custom_hooks = [
    dict(
        type='ConcurrentMLPTrainingHook',
        mlp_epochs=100,                    # Train joint MLP for 100 epochs after each HRNet epoch
        mlp_batch_size=16,                 # MLP batch size
        mlp_lr=1e-5,                       # MLP learning rate (same as standalone training)
        mlp_weight_decay=1e-4,             # MLP weight decay
        hard_example_threshold=5.0,        # MRE threshold for hard-example identification (pixels)
        hrnet_hard_example_weight=2.0,     # Weight multiplier for hard examples in next HRNet epoch (2x oversampling)
        log_interval=20,                   # Log MLP training progress every 20 epochs
        classification_loss_weight=1.0,    # Weight for classification loss
        use_landmark_features_for_classification=True  # Use landmarks for classification
    )
] 