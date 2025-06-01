#!/usr/bin/env python3
"""
Experiment Runner for Cephalometric Landmark Detection
Run different experiments with: python run_experiment.py --index 0
"""

import os
import argparse
import torch
import warnings
import json
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.apis import init_model
from mmengine.runner import Runner
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from experiments_config import experiments

# Suppress warnings
warnings.filterwarnings('ignore')

# Apply PyTorch safe loading fix
import functools
_original_torch_load = torch.load

def safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = safe_torch_load

def plot_training_progress(work_dir):
    """Plot training progress from log files."""
    try:
        import json
        log_file = os.path.join(work_dir, "vis_data", "scalars.json")
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = [json.loads(line) for line in f]
            
            # Extract metrics
            train_loss = [log['loss'] for log in logs if 'loss' in log and log.get('mode') == 'train']
            val_nme = [log['NME'] for log in logs if 'NME' in log and log.get('mode') == 'val']
            
            if train_loss and val_nme:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                ax1.plot(train_loss)
                ax1.set_title('Training Loss')
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('Loss')
                ax1.grid(True)
                
                ax2.plot(val_nme)
                ax2.set_title('Validation NME')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('NME')
                ax2.grid(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join(work_dir, 'training_progress.png'), dpi=150)
                plt.close()
                print(f"Training progress plot saved to {work_dir}/training_progress.png")
    except Exception as e:
        print(f"Could not plot training progress: {e}")

def save_experiment_info(experiment, work_dir):
    """Save experiment configuration to the work directory."""
    info_file = os.path.join(work_dir, 'experiment_info.json')
    with open(info_file, 'w') as f:
        json.dump(experiment, f, indent=2)
    print(f"Experiment info saved to {info_file}")

def generate_config(base_config_path, experiment_config):
    """Generate MMPose configuration from experiment settings."""
    # Load base config
    cfg = Config.fromfile(base_config_path)
    
    # Always start from the same checkpoint
    cfg.load_from = 'Pretrained_model/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth'
    
    # Update codec with experiment's resolution
    cfg.codec = dict(
        type='MSRAHeatmap',
        input_size=experiment_config['input_size'],
        heatmap_size=experiment_config['heatmap_size'],
        sigma=3
    )
    
    # Update model head with loss configuration
    cfg.model.head.out_channels = 19  # Always 19 landmarks
    cfg.model.head.loss = dict(
        type=experiment_config['loss_type'],
        **experiment_config['loss_config']
    )
    
    # Update optimizer
    if experiment_config['optimizer'] == 'Adam':
        cfg.optim_wrapper = dict(
            optimizer=dict(type='Adam', lr=experiment_config['lr']),
            clip_grad=dict(max_norm=5., norm_type=2)
        )
    elif experiment_config['optimizer'] == 'AdamW':
        cfg.optim_wrapper = dict(
            optimizer=dict(
                type='AdamW', 
                lr=experiment_config['lr'],
                weight_decay=experiment_config.get('weight_decay', 0.01)
            ),
            clip_grad=dict(max_norm=5., norm_type=2)
        )
    elif experiment_config['optimizer'] == 'SGD':
        cfg.optim_wrapper = dict(
            optimizer=dict(
                type='SGD', 
                lr=experiment_config['lr'],
                momentum=experiment_config.get('sgd_momentum', 0.9)
            ),
            clip_grad=dict(max_norm=5., norm_type=2)
        )
    
    # Update training config
    cfg.train_cfg = dict(
        by_epoch=True, 
        max_epochs=experiment_config['max_epochs'], 
        val_interval=2
    )
    
    # Update batch size
    cfg.train_dataloader.batch_size = experiment_config['batch_size']
    cfg.val_dataloader.batch_size = experiment_config['batch_size']
    
    # Update augmentation in pipelines
    train_pipeline = [
        dict(type='LoadImageNumpy'),
        dict(type='GetBBoxCenterScale'),
        dict(type='RandomFlip', direction='horizontal'),
        dict(
            type='RandomBBoxTransform',
            shift_prob=0,
            rotate_factor=experiment_config['augmentation']['rotate_factor'],
            scale_factor=experiment_config['augmentation']['scale_factor']
        ),
        dict(type='TopdownAffine', input_size=experiment_config['input_size']),
        dict(type='GenerateTarget', encoder=cfg.codec),
        dict(type='CustomPackPoseInputs', meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape', 'bbox', 'bbox_scores', 'flip_indices', 'center', 'scale', 'input_center', 'input_scale', 'input_size', 'patient_text_id', 'set', 'class'))
    ]
    
    val_pipeline = [
        dict(type='LoadImageNumpy'),
        dict(type='GetBBoxCenterScale'),
        dict(type='TopdownAffine', input_size=experiment_config['input_size']),
        dict(type='CustomPackPoseInputs', meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape', 'bbox', 'bbox_scores', 'flip_indices', 'center', 'scale', 'input_center', 'input_scale', 'input_size', 'patient_text_id', 'set', 'class'))
    ]
    
    cfg.train_dataloader.dataset.pipeline = train_pipeline
    cfg.val_dataloader.dataset.pipeline = val_pipeline
    cfg.test_dataloader.dataset.pipeline = val_pipeline
    
    # Update learning rate scheduler based on optimizer and max_epochs
    cfg.param_scheduler = [
        dict(type='LinearLR', begin=0, end=500, start_factor=1e-3, by_epoch=False),
        dict(
            type='CosineAnnealingLR', 
            T_max=experiment_config['max_epochs'], 
            eta_min=1e-6, 
            by_epoch=True
        )
    ]
    
    return cfg

def main():
    """Main experiment runner function."""
    parser = argparse.ArgumentParser(description='Run cephalometric landmark detection experiments')
    parser.add_argument('--index', type=int, required=True, help='Experiment index to run')
    parser.add_argument('--list', action='store_true', help='List all available experiments')
    args = parser.parse_args()
    
    # List experiments if requested
    if args.list:
        print("\n" + "="*80)
        print("AVAILABLE EXPERIMENTS")
        print("="*80)
        for i, exp in enumerate(experiments):
            print(f"{i}: {exp['name']} - {exp['description']}")
        print("="*80)
        return
    
    # Validate experiment index
    if args.index < 0 or args.index >= len(experiments):
        print(f"Error: Invalid experiment index {args.index}. Valid range: 0-{len(experiments)-1}")
        print("Use --list to see all available experiments")
        return
    
    # Get the experiment
    experiment = experiments[args.index]
    
    print("\n" + "="*80)
    print(f"RUNNING EXPERIMENT {args.index}: {experiment['name']}")
    print("="*80)
    print(f"Description: {experiment['description']}")
    print(f"Configuration:")
    for key, value in experiment['config'].items():
        print(f"  {key}: {value}")
    print("="*80)
    
    # Initialize MMPose scope
    init_default_scope('mmpose')
    
    # Import custom modules
    try:
        import custom_cephalometric_dataset
        import custom_transforms
        import cephalometric_dataset_info
        print("✓ Custom modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import custom modules: {e}")
        return
    
    # Configuration
    base_config_path = "Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py"
    work_dir = f"work_dirs/experiment_{args.index}_{experiment['name']}"
    
    print(f"Base Config: {base_config_path}")
    print(f"Work Dir: {work_dir}")
    
    # Generate configuration for this experiment
    try:
        cfg = generate_config(base_config_path, experiment['config'])
        cfg.work_dir = os.path.abspath(work_dir)
        os.makedirs(cfg.work_dir, exist_ok=True)
        print("✓ Configuration generated successfully")
    except Exception as e:
        print(f"✗ Failed to generate config: {e}")
        return
    
    # Save experiment info
    save_experiment_info(experiment, cfg.work_dir)
    
    # Load and prepare data
    data_file_path = "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json"
    print(f"Loading main data file from: {data_file_path}")
    
    try:
        main_df = pd.read_json(data_file_path)
        print(f"Main DataFrame loaded. Shape: {main_df.shape}")

        train_df = main_df[main_df['set'] == 'train'].reset_index(drop=True)
        val_df = main_df[main_df['set'] == 'dev'].reset_index(drop=True)
        test_df = main_df[main_df['set'] == 'test'].reset_index(drop=True)

        print(f"Train DataFrame shape: {train_df.shape}")
        print(f"Validation DataFrame shape: {val_df.shape}")
        print(f"Test DataFrame shape: {test_df.shape}")

        if train_df.empty or val_df.empty:
            print("ERROR: Training or validation DataFrame is empty.")
            return

        # Save DataFrames to temporary JSON files
        temp_train_ann_file = os.path.join(cfg.work_dir, 'temp_train_ann.json')
        temp_val_ann_file = os.path.join(cfg.work_dir, 'temp_val_ann.json')
        temp_test_ann_file = os.path.join(cfg.work_dir, 'temp_test_ann.json')

        train_df.to_json(temp_train_ann_file, orient='records', indent=2)
        val_df.to_json(temp_val_ann_file, orient='records', indent=2)
        if not test_df.empty:
            test_df.to_json(temp_test_ann_file, orient='records', indent=2)

        # Update config with data files
        cfg.train_dataloader.dataset.ann_file = temp_train_ann_file
        cfg.train_dataloader.dataset.data_df = None
        cfg.train_dataloader.dataset.data_root = ''

        cfg.val_dataloader.dataset.ann_file = temp_val_ann_file
        cfg.val_dataloader.dataset.data_df = None
        cfg.val_dataloader.dataset.data_root = ''

        if not test_df.empty:
            cfg.test_dataloader.dataset.ann_file = temp_test_ann_file
        else:
            cfg.test_dataloader.dataset.ann_file = temp_val_ann_file
        cfg.test_dataloader.dataset.data_df = None
        cfg.test_dataloader.dataset.data_root = ''
        
        print("✓ Configuration updated with data files.")

    except Exception as e:
        print(f"ERROR: Failed to load or process data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print experiment-specific highlights
    print("\n" + "="*60)
    print("EXPERIMENT HIGHLIGHTS")
    print("="*60)
    
    if 'adaptive_wing' in experiment['name']:
        print("🎯 Using AdaptiveWingLoss for robust landmark detection")
        print("   - Better handling of difficult landmarks")
        print("   - Adaptive behavior for different error magnitudes")
    
    if '384x384' in experiment['name']:
        print("📐 High resolution 384×384 for improved precision")
        print("   - 50% increase in resolution")
        print("   - Sub-pixel accuracy improvements")
    elif '512x512' in experiment['name']:
        print("📐 Ultra high resolution 512×512")
        print("   - 100% increase in resolution")
        print("   - Maximum sub-pixel precision")
    
    if 'sgd' in experiment['name']:
        print("⚙️  SGD optimizer with momentum")
        print("   - More stable convergence")
        print("   - Better generalization potential")
    elif 'adamw' in experiment['name']:
        print("⚙️  AdamW optimizer with weight decay")
        print("   - Built-in regularization")
        print("   - Prevents overfitting")
    
    if 'no_augmentation' in experiment['name']:
        print("🔬 No augmentation baseline")
        print("   - Tests raw model capability")
        print("   - Baseline for augmentation effectiveness")
    
    if 'small_batch' in experiment['name']:
        print("📊 Small batch size for better gradients")
        print("   - More frequent updates")
        print("   - Better handling of difficult samples")
    
    if 'low_lr' in experiment['name']:
        print("🐌 Low learning rate with extended training")
        print("   - Fine-grained optimization")
        print("   - Potentially better final convergence")
    
    print("="*60)
    
    # Build runner and start training
    try:
        print("\n🚀 Starting experiment training...")
        runner = Runner.from_cfg(cfg)
        
        print(f"⏱️  Training for {experiment['config']['max_epochs']} epochs...")
        print(f"💾 Results will be saved to: {cfg.work_dir}")
        
        runner.train()
        
        print(f"\n✅ Experiment {args.index} completed successfully!")
        
        # Plot training progress
        plot_training_progress(cfg.work_dir)
        
        # Save final summary
        summary_file = os.path.join(cfg.work_dir, 'experiment_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Experiment: {experiment['name']}\n")
            f.write(f"Description: {experiment['description']}\n")
            f.write(f"Configuration:\n")
            for key, value in experiment['config'].items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nTraining completed successfully.\n")
            f.write(f"Check {cfg.work_dir} for results.\n")
        
        print(f"\n📊 Experiment summary saved to: {summary_file}")
        print(f"🎯 To evaluate results, run:")
        print(f"   python evaluate_detailed_metrics.py --work_dir {cfg.work_dir}")
        
    except Exception as e:
        print(f"\n💥 Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error info
        error_file = os.path.join(cfg.work_dir, 'experiment_error.txt')
        with open(error_file, 'w') as f:
            f.write(f"Experiment: {experiment['name']}\n")
            f.write(f"Error: {str(e)}\n\n")
            import traceback
            f.write(traceback.format_exc())
        print(f"Error details saved to: {error_file}")

if __name__ == "__main__":
    main() 