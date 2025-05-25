import numpy as np
import pandas as pd
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.apis import init_model
import os
import os.path as osp
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Fix for PyTorch 2.6+ weights_only issue
try:
    from mmengine.config.config import ConfigDict
    torch.serialization.add_safe_globals([ConfigDict])
except ImportError:
    pass

# Apply safe torch.load wrapper
import functools
original_torch_load = torch.load

def safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

torch.load = safe_torch_load

def diagnose_training_issues(checkpoint_path: str,
                           config_path: str,
                           data_root: str,
                           test_ann_file: str = 'train_data_pure_old_numpy.json'):
    """
    Comprehensive diagnosis of training issues for cephalometric landmark detection.
    """
    
    print("="*80)
    print("COMPREHENSIVE TRAINING DIAGNOSIS")
    print("="*80)
    
    # Initialize scope
    init_default_scope('mmpose')
    
    # Import custom modules
    try:
        import custom_cephalometric_dataset
        import custom_transforms
        import cephalometric_dataset_info
        from cephalometric_dataset_info import dataset_info, landmark_names_in_order, original_landmark_cols
        print("✓ Custom modules imported successfully")
    except ImportError as e:
        print(f"✗ Custom modules import failed: {e}")
        return
    
    # 1. CONFIG ANALYSIS
    print("\n1. CONFIGURATION ANALYSIS")
    print("-" * 40)
    
    try:
        cfg = Config.fromfile(config_path)
        print(f"✓ Config loaded from: {config_path}")
        
        # Check learning rate
        lr = cfg.optim_wrapper.optimizer.lr
        print(f"Learning Rate: {lr}")
        if lr > 1e-3:
            print(f"⚠️  WARNING: Learning rate {lr} might be too high for fine-tuning!")
            print("   Recommended: 1e-4 to 5e-4 for fine-tuning pretrained models")
        
        # Check optimizer
        optimizer_type = cfg.optim_wrapper.optimizer.type
        print(f"Optimizer: {optimizer_type}")
        
        # Check batch size
        batch_size = cfg.train_dataloader.batch_size
        print(f"Batch Size: {batch_size}")
        
        # Check data preprocessing
        mean = cfg.model.data_preprocessor.mean
        std = cfg.model.data_preprocessor.std
        print(f"Normalization - Mean: {mean}, Std: {std}")
        
        # Check loss function
        loss_type = cfg.model.head.loss.type
        print(f"Loss Function: {loss_type}")
        
        # Check heatmap parameters
        if 'encoder' in cfg.train_dataloader.dataset.pipeline[-2]:
            sigma = cfg.train_dataloader.dataset.pipeline[-2]['encoder']['sigma']
            heatmap_size = cfg.train_dataloader.dataset.pipeline[-2]['encoder']['heatmap_size']
            print(f"Heatmap Sigma: {sigma}, Heatmap Size: {heatmap_size}")
            
            if sigma < 1 or sigma > 3:
                print(f"⚠️  WARNING: Heatmap sigma {sigma} might be inappropriate!")
                print("   Recommended: 1.5-2.5 for 224x224 input images")
        
    except Exception as e:
        print(f"✗ Config analysis failed: {e}")
        return
    
    # 2. MODEL ANALYSIS
    print("\n2. MODEL ANALYSIS")
    print("-" * 40)
    
    try:
        # Load model
        model = init_model(config_path, checkpoint_path, device='cpu')
        print("✓ Model loaded successfully")
        
        # Check if model is properly initialized
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        
        # Check backbone weights (should be from pretrained)
        backbone_weights = model.backbone.state_dict()
        first_conv_weight = list(backbone_weights.values())[0]
        print(f"First Conv Weight Stats - Mean: {first_conv_weight.mean():.6f}, Std: {first_conv_weight.std():.6f}")
        
        # Check head weights (should be randomly initialized or fine-tuned)
        head_weights = model.head.state_dict()
        if 'final_layer.weight' in head_weights:
            final_weight = head_weights['final_layer.weight']
            print(f"Final Layer Weight Stats - Mean: {final_weight.mean():.6f}, Std: {final_weight.std():.6f}")
        
        # Check if head weights are reasonable (not too large/small)
        for name, param in model.head.named_parameters():
            if 'weight' in name:
                weight_mean = param.data.mean().item()
                weight_std = param.data.std().item()
                print(f"Head {name} - Mean: {weight_mean:.6f}, Std: {weight_std:.6f}")
                
                if abs(weight_mean) > 1.0 or weight_std > 1.0:
                    print(f"⚠️  WARNING: {name} has unusual statistics!")
        
    except Exception as e:
        print(f"✗ Model analysis failed: {e}")
        return
    
    # 3. DATA ANALYSIS
    print("\n3. DATA ANALYSIS")
    print("-" * 40)
    
    try:
        # Load and analyze dataset
        data_path = osp.join(data_root, test_ann_file)
        df = pd.read_json(data_path)
        print(f"✓ Dataset loaded: {len(df)} total samples")
        
        # Check data distribution
        if 'set' in df.columns:
            print("Data split distribution:")
            print(df['set'].value_counts())
        
        # Analyze coordinate ranges
        landmark_cols = original_landmark_cols
        x_cols = landmark_cols[::2]  # Every other starting from 0
        y_cols = landmark_cols[1::2]  # Every other starting from 1
        
        print("\nCoordinate Analysis:")
        for i, (x_col, y_col) in enumerate(zip(x_cols, y_cols)):
            if x_col in df.columns and y_col in df.columns:
                x_vals = df[x_col].dropna()
                y_vals = df[y_col].dropna()
                
                print(f"  {landmark_names_in_order[i]:20s}: X[{x_vals.min():.1f}, {x_vals.max():.1f}], Y[{y_vals.min():.1f}, {y_vals.max():.1f}]")
                
                # Check for suspicious coordinates
                if x_vals.min() < 0 or y_vals.min() < 0:
                    print(f"    ⚠️  WARNING: Negative coordinates found!")
                if x_vals.max() > 224 or y_vals.max() > 224:
                    print(f"    ⚠️  WARNING: Coordinates exceed image bounds (224x224)!")
        
        # Check for missing landmarks
        missing_counts = {}
        for i, (x_col, y_col) in enumerate(zip(x_cols, y_cols)):
            if x_col in df.columns and y_col in df.columns:
                missing = df[x_col].isna() | df[y_col].isna() | (df[x_col] == 0) | (df[y_col] == 0)
                missing_count = missing.sum()
                missing_counts[landmark_names_in_order[i]] = missing_count
                
                if missing_count > 0:
                    print(f"  {landmark_names_in_order[i]:20s}: {missing_count}/{len(df)} missing ({100*missing_count/len(df):.1f}%)")
        
        # Check most problematic landmarks
        if missing_counts:
            worst_landmarks = sorted(missing_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"\nMost missing landmarks: {[f'{name}({count})' for name, count in worst_landmarks]}")
    
    except Exception as e:
        print(f"✗ Data analysis failed: {e}")
        return
    
    # 4. HEATMAP TARGET ANALYSIS
    print("\n4. HEATMAP TARGET ANALYSIS")
    print("-" * 40)
    
    try:
        # Create a sample heatmap to check target generation
        from mmpose.codecs import MSRAHeatmap
        
        # Get a sample from the dataset
        test_df = df[df['set'] == 'test'].head(1) if 'set' in df.columns else df.head(1)
        sample = test_df.iloc[0]
        
        # Extract sample keypoints
        keypoints = np.zeros((19, 2))
        for i, (x_col, y_col) in enumerate(zip(x_cols, y_cols)):
            if x_col in sample and y_col in sample and pd.notna(sample[x_col]) and pd.notna(sample[y_col]):
                keypoints[i, 0] = sample[x_col]
                keypoints[i, 1] = sample[y_col]
        
        print(f"Sample keypoints shape: {keypoints.shape}")
        print(f"Sample keypoints range: X[{keypoints[:, 0].min():.1f}, {keypoints[:, 0].max():.1f}], Y[{keypoints[:, 1].min():.1f}, {keypoints[:, 1].max():.1f}]")
        
        # Generate heatmap
        encoder = MSRAHeatmap(
            input_size=(224, 224),
            heatmap_size=(56, 56),
            sigma=2
        )
        
        # Check heatmap generation
        encoded = encoder.encode(keypoints, (224, 224))
        heatmaps = encoded['heatmaps']
        
        print(f"Generated heatmap shape: {heatmaps.shape}")
        print(f"Heatmap value range: [{heatmaps.min():.4f}, {heatmaps.max():.4f}]")
        
        # Count non-zero heatmaps
        non_zero_maps = (heatmaps.max(axis=(1, 2)) > 0.01).sum()
        print(f"Non-zero heatmaps: {non_zero_maps}/{len(landmark_names_in_order)}")
        
        if non_zero_maps < len(landmark_names_in_order) // 2:
            print("⚠️  WARNING: More than half of heatmaps are empty!")
            print("   This might indicate coordinate issues or inappropriate sigma")
    
    except Exception as e:
        print(f"✗ Heatmap analysis failed: {e}")
    
    # 5. RECOMMENDED FIXES
    print("\n5. RECOMMENDED FIXES")
    print("-" * 40)
    
    print("Based on the analysis, here are the recommended fixes:")
    print()
    
    print("1. LEARNING RATE ADJUSTMENT:")
    print("   - Current LR might be too high, try: 1e-4 or 2e-4")
    print("   - Use cosine annealing instead of step decay")
    print()
    
    print("2. HEATMAP PARAMETERS:")
    print("   - Increase sigma to 2.5-3.0 for better target coverage")
    print("   - Verify coordinate normalization is correct")
    print()
    
    print("3. MISSING LANDMARKS:")
    print("   - Handle missing Gonion landmarks properly in loss calculation")
    print("   - Use weighted loss to handle imbalanced landmarks")
    print()
    
    print("4. DATA PREPROCESSING:")
    print("   - Verify image normalization matches pretrained model")
    print("   - Check if coordinate scaling is applied correctly")
    print()
    
    print("5. TRAINING STRATEGY:")
    print("   - Start with frozen backbone for few epochs")
    print("   - Use smaller batch size (16) for more stable gradients")
    print("   - Add validation monitoring to detect overfitting")

def create_fixed_config(original_config_path: str, output_path: str):
    """Create a fixed configuration with recommended improvements."""
    
    print(f"\nCreating fixed config at: {output_path}")
    
    # Read original config
    cfg = Config.fromfile(original_config_path)
    
    # Apply fixes
    # 1. Lower learning rate
    cfg.optim_wrapper.optimizer.lr = 2e-4
    
    # 2. Improved scheduler
    cfg.param_scheduler = [
        dict(
            type='LinearLR',
            start_factor=0.001,
            by_epoch=False,
            begin=0,
            end=500
        ),
        dict(
            type='CosineAnnealingLR',
            begin=0,
            end=60,
            by_epoch=True,
            T_max=60,
            eta_min=1e-6
        )
    ]
    
    # 3. Smaller batch size
    cfg.train_dataloader.batch_size = 16
    
    # 4. Improved heatmap sigma
    for i, transform in enumerate(cfg.train_dataloader.dataset.pipeline):
        if transform.get('type') == 'GenerateTarget':
            cfg.train_dataloader.dataset.pipeline[i]['encoder']['sigma'] = 2.5
    
    # 5. Add validation
    cfg.val_dataloader = dict(
        batch_size=16,
        num_workers=2,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='CustomCephalometricDataset',
            data_root=cfg.train_dataloader.dataset.data_root,
            ann_file='train_data_pure_old_numpy.json',
            metainfo=cfg.train_dataloader.dataset.metainfo,
            pipeline=cfg.val_pipeline,
            test_mode=True,
        )
    )
    
    cfg.val_evaluator = dict(
        type='PCKAccuracy',
        thr=0.05,  # 5% of image size
        norm_item=['bbox', 'torso']
    )
    
    cfg.train_cfg.val_interval = 5
    
    # Save fixed config
    cfg.dump(output_path)
    print("✓ Fixed config saved!")

if __name__ == "__main__":
    # Run diagnosis
    checkpoint_path = "/content/ceph_mmpose_gemini/work_dirs/hrnetv2_w18_cephalometric_experiment/epoch_2.pth"
    config_path = "/content/ceph_mmpose_gemini/configs/hrnetv2/hrnetv2_w18_cephalometric_224x224.py"
    data_root = "/content/drive/MyDrive/Lala's Masters/"
    
    diagnose_training_issues(checkpoint_path, config_path, data_root)
    
    # Create fixed config
    fixed_config_path = "/content/ceph_mmpose_gemini/configs/hrnetv2/hrnetv2_w18_cephalometric_224x224_FIXED.py"
    create_fixed_config(config_path, fixed_config_path) 