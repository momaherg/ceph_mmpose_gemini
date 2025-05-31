_base_ = ['./td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py'] # Inherit from original

# Fine-tuning specific
load_from = 'Pretrained_model/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth'
train_cfg = dict(by_epoch=True, max_epochs=30, val_interval=1) # Adjusted for fine-tuning

# Optimizer settings for fine-tuning (overrides base config's optim_wrapper)
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=5e-5), # Smaller learning rate for fine-tuning
    # clip_grad=None # Explicitly set if you want to ensure no gradient clipping or inherit from base
)

# Dataset settings
dataset_type = 'CustomCephalometricDataset' # Your custom dataset
data_root = "/content/drive/MyDrive/Lala\'s Masters/" # Conventional data root, actual data comes from data_df injected by training script
# ann_file_main = 'train_data_pure_old_numpy.json' # The single JSON file

# Codec (should match pretraining, especially input_size for TopdownAffine)
codec = dict(
    type='MSRAHeatmap',
    input_size=(256, 256), # Crucial: Model expects 256x256 input
    heatmap_size=(64, 64),
    sigma=2)

# Model head output channels should match number of keypoints (19 for your dataset)
model = dict(
    head=dict(out_channels=19), # Ensure this matches your dataset's keypoint count
    # The rest of the model (backbone, neck, data_preprocessor, test_cfg)
    # can be inherited or slightly adjusted if needed.
    # data_preprocessor mean/std are from ImageNet, generally fine for transfer.
)


# Pipelines - Remove LoadImage, ensure TopdownAffine targets 256x256
train_pipeline = [
    # dict(type='LoadImage'), # REMOVED: Custom dataset loads image array directly
    dict(type='GetBBoxCenterScale'), # Uses bbox from custom dataset
    dict(type='RandomFlip', direction='horizontal'),
    dict(
        type='RandomBBoxTransform',
        shift_prob=0, # AFLW config had this at 0
        rotate_factor=60,
        scale_factor=(0.75, 1.25)),
    dict(type='TopdownAffine', input_size=codec['input_size']), # Use 256x256
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs', meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape', 'bbox', 'bbox_scores', 'flip_indices', 'center', 'scale', 'input_center', 'input_scale', 'input_size', 'patient_text_id', 'set', 'class'))
]
val_pipeline = [
    # dict(type='LoadImage'), # REMOVED
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']), # Use 256x256
    dict(type='PackPoseInputs', meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape', 'bbox', 'bbox_scores', 'flip_indices', 'center', 'scale', 'input_center', 'input_scale', 'input_size', 'patient_text_id', 'set', 'class'))
]
test_pipeline = val_pipeline # Test pipeline often same as validation

# DataLoaders
# The CustomCephalometricDataset needs to be able to split the data from the single JSON
# or the training script needs to prepare and pass pandas DataFrames (train_df, val_df, test_df)
# to each dataloader's dataset config using the `data_df` argument.

# Option 1: Using ann_file and hoping CustomCephalometricDataset filters by 'set'
# This is simpler if the dataset supports it. If not, Option 2 is needed.

# Option 2: Modify training script to inject data_df.
# Example structure if data_df is injected by training script:
# (ann_file and data_root might become '' or None if data_df is primary)

train_dataloader = dict(
    batch_size=32, # Adjust as per your GPU memory, original was 64
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
    batch_size=32, # Adjust as per your GPU memory
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

# Evaluators
val_evaluator = dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=list(range(19))) # Adjust keypoint_indices if needed
test_evaluator = val_evaluator

# You might want to adjust learning rate for fine-tuning
# optim_wrapper = dict(optimizer=dict(lr=5e-5)) # Example: smaller LR # This line is now redundant and can be removed

# Visualization (optional, but good for debugging)
# visualizer = dict(
#    type='PoseLocalVisualizer',
#    vis_backends=[dict(type='LocalVisBackend')],
#    name='visualizer')
# default_hooks = dict(
#    visualization=dict(enable=True, type='PoseVisualizationHook')) 