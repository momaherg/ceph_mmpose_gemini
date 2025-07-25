_base_ = ['./td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py'] # Inherit from original

# Fine-tuning specific
load_from = 'Pretrained_model/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth'
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=2) # Extended training: 100 -> 100 epochs

# Improved optimizer settings with cosine annealing
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=3e-4), # Higher starting LR for cosine schedule
    clip_grad=dict(max_norm=5.,  # see next section
                   norm_type=2)
)

# Learning rate scheduler with warm-up and step decay
param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, start_factor=1e-3, by_epoch=False),  # Warm-up
    # dict(type='CosineAnnealingLR', T_max=100, eta_min=1e-6, by_epoch=True)  # Cosine annealing - updated T_max to match max_epochs
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
# ann_file_main = 'train_data_pure_old_numpy.json' # The single JSON file

# Codec - UPGRADED: Higher resolution for better precision
codec = dict(
    type='MSRAHeatmap',
    input_size=(384, 384), # UPGRADED: Was (256, 256) - Higher resolution for finer details
    heatmap_size=(96, 96),  # UPGRADED: Was (64, 64) - Larger heatmaps for sub-pixel precision
    sigma=3)

# Model with Classification Head
model = dict(
    # Ensure we keep the neck from base config
    neck=dict(concat=True, type='FeatureMapProcessor'),
    head=dict(
        type='HRNetV2WithClassificationSimple',  # Use our simpler custom head
        in_channels=270,  # Important: HRNet with neck outputs 270 channels (18+36+72+144)
        out_channels=19, # Number of keypoints
        # Conv layer parameters from base config
        conv_out_channels=(270,),  # Single conv layer with 270 channels
        conv_kernel_sizes=(1,),    # 1x1 convolution
        deconv_out_channels=None,  # No deconv layers in this config
        # Classification head parameters
        num_classes=3,  # Skeletal Class I, II, III
        classification_hidden_dim=256,
        classification_dropout=0.2,
        classification_loss_weight=2.0,  # Increased weight for better classification learning
        # Class weights based on actual distribution (Class I=152, II=77, III=22)
        class_weights=[0.55, 1.08, 3.79],  # Balanced weights for imbalanced dataset
        # Keypoint detection loss
        loss=dict(  # main loss
                type='AdaptiveWingLoss',
                alpha=2.1,  omega=24., epsilon=1., theta=0.5,
                use_target_weight=False, loss_weight=1.0),
        # Decoder for heatmap
        decoder=codec  # Use the same codec defined above
    )
    # The rest of the model (backbone, neck, data_preprocessor, test_cfg)
    # can be inherited or slightly adjusted if needed.
    # data_preprocessor mean/std are from ImageNet, generally fine for transfer.
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
    dict(type='CustomPackPoseInputs', meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape', 'bbox', 'bbox_scores', 'flip_indices', 'center', 'scale', 'input_center', 'input_scale', 'input_size', 'patient_text_id', 'set', 'class', 'gt_classification'))
]
val_pipeline = [
    dict(type='LoadImageNumpy'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']), # Now uses 384x384
    dict(type='CustomPackPoseInputs', meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape', 'bbox', 'bbox_scores', 'flip_indices', 'center', 'scale', 'input_center', 'input_scale', 'input_size', 'patient_text_id', 'set', 'class', 'gt_classification'))
]
test_pipeline = val_pipeline # Test pipeline often same as validation

# DataLoaders - REDUCED batch size for higher resolution
# The CustomCephalometricDataset needs to be able to split the data from the single JSON
# or the training script needs to prepare and pass pandas DataFrames (train_df, val_df, test_df)
# to each dataloader's dataset config using the `data_df` argument.

# Option 1: Using ann_file and hoping CustomCephalometricDataset filters by 'set'
# This is simpler if the dataset supports it. If not, Option 2 is needed.

# Option 2: Modify training script to inject data_df.
# Example structure if data_df is injected by training script:
# (ann_file and data_root might become '' or None if data_df is primary)

train_dataloader = dict(
    batch_size=20, # REDUCED: Was 32 - Lower for 384x384 resolution to manage GPU memory
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        # data_root=data_root, # Base path for ann_file if it's relative
        # ann_file=ann_file_main, # Path to the JSON file with all data
        # data_df=None, # This would be replaced by the actual train_df by the training script
        data_mode='topdown',
        pipeline=train_pipeline,
        # The CustomCephalometricDataset's METAINFO will be used.
        # If your CustomCephalometricDataset doesn't filter internally based on 'set' from ann_file,
        # you MUST provide a pre-filtered `data_df` here via the training script.
        # For demonstration, we assume the training script will pass a `train_df` to `data_df`.
        # If you want the dataset to load and filter, it needs that logic.
        # For now, we'll leave ann_file commented out, assuming data_df is the primary way for this fine-tuning setup.
        ann_file='', # To be populated by filtered data_df or if dataset handles split from one file
        # test_mode=False # Explicitly false for training
    ))

val_dataloader = dict(
    batch_size=20, # REDUCED: Was 32 - Lower for 384x384 resolution to manage GPU memory
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        # data_root=data_root,
        # ann_file=ann_file_main,
        # data_df=None, # This would be replaced by the actual val_df by the training script
        data_mode='topdown',
        pipeline=val_pipeline,
        ann_file='',
        test_mode=True # Crucial for validation
    ))
test_dataloader = val_dataloader # Often, test and val dataloaders are configured similarly

# Evaluators - Add classification metrics
val_evaluator = [
    dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=[0, 1]),  # Keypoint evaluation
    dict(type='ClassificationMetric', num_classes=3)  # Classification evaluation
]
test_evaluator = val_evaluator

# Visualization (optional, but good for debugging)
# visualizer = dict(
#    type='PoseLocalVisualizer',
#    vis_backends=[dict(type='LocalVisBackend')],
#    name='visualizer')
# default_hooks = dict(
#    visualization=dict(enable=True, type='PoseVisualizationHook')) 

# =========================================================================
# CONCURRENT JOINT MLP TRAINING HOOK WITH HRNET HARD-EXAMPLE OVERSAMPLING
# =========================================================================
# This hook trains a joint MLP refinement model on-the-fly during HRNetV2 training.
# After each HRNet epoch, it:
# 1. Runs inference on training data using current HRNet weights
# 2. Identifies hard examples based on landmark prediction errors
# 3. Trains a joint 38-D MLP model for 100 epochs with hard-example oversampling
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
        log_interval=20                    # Log MLP training progress every 20 epochs
    )
] 