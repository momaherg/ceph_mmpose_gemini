"""
Experiment configurations for cephalometric landmark detection.
Each experiment starts from the same pre-trained checkpoint and explores different hyperparameters.
"""

experiments = [
    # Experiment 0: Baseline (256x256, MSE Loss)
    {
        "name": "baseline_256x256_mse",
        "description": "Baseline configuration with 256x256 resolution and MSE loss",
        "config": {
            "input_size": (256, 256),
            "heatmap_size": (64, 64),
            "loss_type": "KeypointMSELoss",
            "loss_config": {},
            "optimizer": "Adam",
            "lr": 5e-4,
            "batch_size": 32,
            "max_epochs": 50,
            "augmentation": {
                "rotate_factor": 15,
                "scale_factor": (0.85, 1.15),
            }
        }
    },
    
    # Experiment 1: Higher Resolution (384x384, MSE Loss)
    {
        "name": "highres_384x384_mse",
        "description": "Higher resolution 384x384 with MSE loss",
        "config": {
            "input_size": (384, 384),
            "heatmap_size": (96, 96),
            "loss_type": "KeypointMSELoss",
            "loss_config": {},
            "optimizer": "Adam",
            "lr": 5e-4,
            "batch_size": 20,
            "max_epochs": 60,
            "augmentation": {
                "rotate_factor": 20,
                "scale_factor": (0.8, 1.2),
            }
        }
    },
    
    # Experiment 2: AdaptiveWingLoss with 256x256
    {
        "name": "adaptive_wing_256x256",
        "description": "AdaptiveWingLoss with standard 256x256 resolution",
        "config": {
            "input_size": (256, 256),
            "heatmap_size": (64, 64),
            "loss_type": "AdaptiveWingLoss",
            "loss_config": {
                "alpha": 2.1,
                "omega": 24.0,
                "epsilon": 1.0,
                "theta": 0.5,
                "use_target_weight": False
            },
            "optimizer": "Adam",
            "lr": 3e-4,
            "batch_size": 32,
            "max_epochs": 60,
            "augmentation": {
                "rotate_factor": 30,
                "scale_factor": (0.7, 1.3),
            }
        }
    },
    
    # Experiment 3: AdaptiveWingLoss with 384x384 (Full upgrade)
    {
        "name": "adaptive_wing_384x384",
        "description": "AdaptiveWingLoss with high resolution 384x384",
        "config": {
            "input_size": (384, 384),
            "heatmap_size": (96, 96),
            "loss_type": "AdaptiveWingLoss",
            "loss_config": {
                "alpha": 2.1,
                "omega": 24.0,
                "epsilon": 1.0,
                "theta": 0.5,
                "use_target_weight": False
            },
            "optimizer": "Adam",
            "lr": 3e-4,
            "batch_size": 20,
            "max_epochs": 60,
            "augmentation": {
                "rotate_factor": 30,
                "scale_factor": (0.7, 1.3),
            }
        }
    },
    
    # Experiment 4: SGD Optimizer with momentum
    {
        "name": "sgd_momentum_256x256",
        "description": "SGD with momentum optimizer",
        "config": {
            "input_size": (256, 256),
            "heatmap_size": (64, 64),
            "loss_type": "KeypointMSELoss",
            "loss_config": {},
            "optimizer": "SGD",
            "lr": 1e-3,
            "sgd_momentum": 0.9,
            "batch_size": 32,
            "max_epochs": 80,
            "augmentation": {
                "rotate_factor": 15,
                "scale_factor": (0.85, 1.15),
            }
        }
    },
    
    # Experiment 5: Very low learning rate with long training
    {
        "name": "low_lr_long_train",
        "description": "Very low learning rate with extended training",
        "config": {
            "input_size": (256, 256),
            "heatmap_size": (64, 64),
            "loss_type": "KeypointMSELoss",
            "loss_config": {},
            "optimizer": "Adam",
            "lr": 1e-4,
            "batch_size": 32,
            "max_epochs": 100,
            "augmentation": {
                "rotate_factor": 20,
                "scale_factor": (0.8, 1.2),
            }
        }
    },
    
    # Experiment 6: AdamW optimizer with weight decay
    {
        "name": "adamw_weight_decay",
        "description": "AdamW optimizer with weight decay regularization",
        "config": {
            "input_size": (256, 256),
            "heatmap_size": (64, 64),
            "loss_type": "AdaptiveWingLoss",
            "loss_config": {
                "alpha": 2.1,
                "omega": 24.0,
                "epsilon": 1.0,
                "theta": 0.5,
                "use_target_weight": False
            },
            "optimizer": "AdamW",
            "lr": 5e-4,
            "weight_decay": 0.01,
            "batch_size": 32,
            "max_epochs": 60,
            "augmentation": {
                "rotate_factor": 25,
                "scale_factor": (0.75, 1.25),
            }
        }
    },
    
    # Experiment 7: Smaller batch size for better gradients
    {
        "name": "small_batch_adaptive",
        "description": "Smaller batch size with adaptive wing loss",
        "config": {
            "input_size": (256, 256),
            "heatmap_size": (64, 64),
            "loss_type": "AdaptiveWingLoss",
            "loss_config": {
                "alpha": 2.1,
                "omega": 24.0,
                "epsilon": 1.0,
                "theta": 0.5,
                "use_target_weight": False
            },
            "optimizer": "Adam",
            "lr": 2e-4,
            "batch_size": 16,
            "max_epochs": 60,
            "augmentation": {
                "rotate_factor": 30,
                "scale_factor": (0.7, 1.3),
            }
        }
    },
    
    # Experiment 8: No augmentation baseline
    {
        "name": "no_augmentation",
        "description": "Baseline without data augmentation",
        "config": {
            "input_size": (256, 256),
            "heatmap_size": (64, 64),
            "loss_type": "KeypointMSELoss",
            "loss_config": {},
            "optimizer": "Adam",
            "lr": 5e-4,
            "batch_size": 32,
            "max_epochs": 50,
            "augmentation": {
                "rotate_factor": 0,
                "scale_factor": (1.0, 1.0),
            }
        }
    },
    
    # Experiment 9: Ultra high resolution (512x512)
    {
        "name": "ultra_highres_512x512",
        "description": "Ultra high resolution 512x512 experiment",
        "config": {
            "input_size": (512, 512),
            "heatmap_size": (128, 128),
            "loss_type": "AdaptiveWingLoss",
            "loss_config": {
                "alpha": 2.1,
                "omega": 32.0,  # Increased for larger heatmaps
                "epsilon": 1.0,
                "theta": 0.5,
                "use_target_weight": False
            },
            "optimizer": "Adam",
            "lr": 2e-4,
            "batch_size": 12,  # Reduced for memory
            "max_epochs": 60,
            "augmentation": {
                "rotate_factor": 30,
                "scale_factor": (0.7, 1.3),
            }
        }
    }
] 