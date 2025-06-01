_base_ = './hrnetv2_w18_cephalometric_256x256_finetune.py' # Inherits 384x384 setup, sigma=3

# Experiment 3: 384x384 + MLECCLoss

# Model: MLECCLoss
model = dict(
    head=dict(
        loss=dict(
            type='MLECCLoss',
            use_target_weight=True, # Crucial for joint_weights to have an effect
            # alpha=2.0, beta=2.0, # Default values, can be omitted
            loss_weight=1.0
        )
    )
)

# Optimizer: LR for this loss
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-4),
    clip_grad=dict(max_norm=5., norm_type=2)
)

# Training schedule: 60 epochs (as per base)
train_cfg = dict(by_epoch=True, max_epochs=60, val_interval=2)

# LR scheduler: (as per base for 60 epochs)
param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, start_factor=1e-3, by_epoch=False),
    dict(type='CosineAnnealingLR', T_max=60, eta_min=1e-6, by_epoch=True)
]

# Enable flip test for MLECCLoss, as it can benefit
test_cfg = dict(
    flip_test=True,
    # other test_cfg from base will be inherited (flip_mode, shift_heatmap)
) 