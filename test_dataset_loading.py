#!/usr/bin/env python3
"""
Test script to verify dataset loading and HRNetV2 prediction extraction.
This helps debug issues before running the full training pipeline.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append('.')

def test_dataset_loading():
    """Test the dataset loading with various configurations."""
    
    print("="*80)
    print("TESTING MLP REFINEMENT DATASET LOADING")
    print("="*80)
    
    try:
        import torch
        import pandas as pd
        from mmengine.registry import init_default_scope
        
        # Initialize MMPose scope
        init_default_scope('mmpose')
        
        # Import custom modules
        import custom_cephalometric_dataset
        import custom_transforms
        from mlp_refinement_dataset import MLPRefinementDataset
        
        print("✓ All imports successful")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running in the correct environment with mmpose installed")
        return False
    
    # Configuration
    config = {
        'data_file': "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json",
        'hrnet_config': "Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py",
        'hrnet_checkpoint': "work_dirs/hrnetv2_w18_cephalometric_384x384_adaptive_wing_loss_v4/epoch_5.pth",
        'input_size': 384,
    }
    
    # Check files exist
    print("\n📁 Checking files...")
    for key, path in config.items():
        if key.endswith('_file') or key.endswith('_config') or key.endswith('_checkpoint'):
            if os.path.exists(path):
                print(f"✓ {key}: {path}")
            else:
                print(f"❌ {key}: {path} - NOT FOUND")
                return False
    
    # Load data
    print("\n📊 Loading data...")
    try:
        main_df = pd.read_json(config['data_file'])
        test_df = main_df[main_df['set'] == 'dev'].head(10).reset_index(drop=True)  # Small test set
        print(f"✓ Loaded {len(test_df)} test samples")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    # Test CPU inference
    print("\n🧠 Testing CPU inference...")
    try:
        dataset_cpu = MLPRefinementDataset(
            test_df, 
            config['hrnet_config'], 
            config['hrnet_checkpoint'],
            input_size=config['input_size'], 
            cache_predictions=True,
            force_cpu=True
        )
        print("✓ CPU dataset creation successful")
        
        # Test loading a sample
        print("\n🎯 Testing sample loading...")
        sample = dataset_cpu[0]
        print(f"✓ Sample loaded successfully")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  HRNet predictions shape: {sample['hrnet_predictions'].shape}")
        print(f"  Ground truth shape: {sample['ground_truth'].shape}")
        print(f"  Valid mask shape: {sample['valid_mask'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during CPU testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataset_loading()
    
    if success:
        print("\n" + "="*80)
        print("✅ DATASET TESTING COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nYou can now run the full training script with confidence!")
        print("Next steps:")
        print("  1. Run: python3 train_mlp_refinement.py")
        print("  2. Monitor the training progress")
        print("  3. Evaluate results with: python3 evaluate_mlp_refinement.py")
    else:
        print("\n" + "="*80)
        print("❌ DATASET TESTING FAILED")
        print("="*80)
        print("\nPlease fix the issues above before running the training script.")
        print("Common solutions:")
        print("  - Check file paths are correct")
        print("  - Ensure mmpose environment is properly set up")
        print("  - Try running with CUDA_LAUNCH_BLOCKING=1 for more detailed errors") 