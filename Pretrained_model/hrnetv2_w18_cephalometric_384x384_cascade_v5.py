# Inherit from the original fine-tuning config to get all the dataset and pipeline settings
_base_ = ['./hrnetv2_w18_cephalometric_256x256_finetune.py']

# -- MODEL CONFIGURATION: LAYER 2 CASCADE ---
# Point to the best checkpoint from the previous run to initialize Stage 1
load_from = '/content/ceph_mmpose_gemini/work_dirs/hrnetv2_w18_cephalometric_384x384_adaptive_wing_loss_v4/best_NME_epoch_86.pth' # Using the best V4 checkpoint

# Define the custom model type from our new file
custom_imports = dict(imports=['refinement_hrnet'], allow_failed_imports=False)

# Build the two-stage cascade model
model = dict(
    type='RefinementHRNet', # Our new custom model class
    # The backbone, neck, and data_preprocessor are inherited from _base_
    # We only need to redefine the head structure
    head=dict(
        # This is Stage 1, the original heatmap head
        type='HeatmapHead',
        in_channels=18, # from HRNetv2-w18
        out_channels=19,
        loss=dict(
            type='AdaptiveWingLoss',
            use_target_weight=True,
            loss_weight=1.0)),
    refine_head=dict(
        # This is Stage 2, a small regression head for refinement
        type='RegressionHead',
        in_channels=18, # from HRNetv2-w18 features
        num_joints=19,
        loss=dict(type='MSELoss', use_target_weight=True, loss_weight=0.2), # Lower weight for the refinement loss
        decoder=dict(
            type='RegressionLabel',
            input_size=(32, 32), # patch size
            output_size=(32, 32), # Does not matter for regression
        )
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
        # flip_indices are inherited from dataset metaifo
    )
)

# --- TRAINING SETTINGS FOR REFINEMENT ---
# We are fine-tuning the refinement head, so we can use fewer epochs and a lower learning rate
train_cfg = dict(by_epoch=True, max_epochs=40, val_interval=1)

# Lower learning rate for fine-tuning
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-4))

param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict(
        type='MultiStepLR',
        begin=0,
        end=40,
        by_epoch=True,
        milestones=[20, 35],
        gamma=0.1)
]

# Optional: Freeze the backbone for the first few epochs to stabilize refinement head training
# paramwise_cfg = dict(
#     custom_keys={
#         'backbone': dict(lr_mult=0.0, decay_mult=0.0),
#     }
# )
# resume=True # Resume from the checkpoint specified in load_from 