# Inherit from the previous fine-tuned model config
_base_ = ['./hrnetv2_w18_cephalometric_256x256_finetune.py']

# --- STAGE 2: CASCADE REFINEMENT CONFIG (V5) ---

# Load from the best checkpoint of the previous (V4) training
# IMPORTANT: Update this path to your actual best checkpoint from the v4 experiment
load_from = "/content/ceph_mmpose_gemini/work_dirs/hrnetv2_w18_cephalometric_384x384_adaptive_wing_loss_v4/best_NME_epoch_86.pth"

# Fine-tune for fewer epochs as the backbone is already strong
train_cfg = dict(by_epoch=True, max_epochs=40, val_interval=1)

# Use a smaller learning rate for fine-tuning the new head
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-4))

# Fine-tuning learning rate schedule
param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, start_factor=0.01, by_epoch=False),
    dict(type='CosineAnnealingLR', T_max=40, eta_min=1e-7, by_epoch=True)
]

# --- MODEL DEFINITION: RefinementHRNet ---
model = dict(
    type='RefinementHRNet',  # Use our custom cascade model
    # The backbone, neck, and base head are inherited from the base config
    # and will have their weights loaded from the `load_from` checkpoint.

    # --- Refinement Head (Stage 2) ---
    refine_head=dict(
        type='RegressionHead',
        in_channels=18,  # HRNetV2-w18 has 18 channels at the highest resolution feature map
        num_joints=19,
        loss=dict(type='MSELoss', use_target_weight=True, loss_weight=1.0),
        decoder=dict(
            type='RegressionLabel',
            # This is the input size of the *feature patch* given to the refine head.
            # It must match `patch_size` in the `RefinementHRNet` model implementation.
            # This fixes the TypeError.
            input_size=(32, 32),
        )
    )
)

# Keep the same datasets, pipelines, and evaluators from the base config 