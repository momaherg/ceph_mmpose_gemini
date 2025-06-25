#!/usr/bin/env python3
"""
Verification Script for Concurrent MLP Training Setup
"""

import os
import sys
import importlib.util

def check_file_exists(filepath, description):
    """Check if a file exists and report status."""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} (NOT FOUND)")
        return False

def check_import(module_name, description):
    """Check if a module can be imported."""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            print(f"✓ {description}: Can import {module_name}")
            return True
        else:
            print(f"✗ {description}: Cannot find {module_name}")
            return False
    except Exception as e:
        print(f"✗ {description}: Import error for {module_name}: {e}")
        return False

def main():
    """Main verification function."""
    print("="*70)
    print("CONCURRENT MLP TRAINING SETUP VERIFICATION")
    print("="*70)
    
    all_good = True
    
    # Check essential files
    print("\n📁 Essential Files:")
    files_to_check = [
        ("mlp_concurrent_training_hook.py", "MLP Training Hook"),
        ("train_concurrent_v5.py", "Concurrent Training Script"),
        ("Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py", "Config File"),
        ("custom_cephalometric_dataset.py", "Custom Dataset"),
        ("custom_transforms.py", "Custom Transforms"),
        ("cephalometric_dataset_info.py", "Dataset Info"),
    ]
    
    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_good = False
    
    # Check if hook can be imported
    print("\n🔧 Module Imports:")
    try:
        sys.path.insert(0, os.getcwd())
        import mlp_concurrent_training_hook
        print("✓ Hook Import: mlp_concurrent_training_hook imported successfully")
        
        # Check if hook is properly registered
        if hasattr(mlp_concurrent_training_hook, 'ConcurrentMLPTrainingHook'):
            print("✓ Hook Class: ConcurrentMLPTrainingHook found")
        else:
            print("✗ Hook Class: ConcurrentMLPTrainingHook not found")
            all_good = False
            
    except Exception as e:
        print(f"✗ Hook Import: Failed to import mlp_concurrent_training_hook: {e}")
        all_good = False
    
    # Check MMPose dependencies
    dependencies = [
        ("mmengine", "MMEngine"),
        ("mmpose", "MMPose"),
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
    ]
    
    print("\n📦 Dependencies:")
    for module_name, description in dependencies:
        if not check_import(module_name, description):
            all_good = False
    
    # Check config file content
    print("\n⚙️  Configuration:")
    try:
        from mmengine.config import Config
        cfg = Config.fromfile("Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py")
        
        if hasattr(cfg, 'custom_hooks'):
            hooks = cfg.custom_hooks
            print(f"✓ Custom Hooks: {len(hooks)} hook(s) configured")
            
            # Check for our specific hook
            concurrent_hook_found = False
            for hook in hooks:
                if hook.get('type') == 'ConcurrentMLPTrainingHook':
                    concurrent_hook_found = True
                    print(f"✓ Hook Config: ConcurrentMLPTrainingHook found")
                    print(f"   - MLP epochs: {hook.get('mlp_epochs', 'not set')}")
                    print(f"   - MLP batch size: {hook.get('mlp_batch_size', 'not set')}")
                    print(f"   - MLP learning rate: {hook.get('mlp_lr', 'not set')}")
                    break
            
            if not concurrent_hook_found:
                print("✗ Hook Config: ConcurrentMLPTrainingHook not found in config")
                all_good = False
        else:
            print("✗ Custom Hooks: No custom_hooks found in config")
            all_good = False
            
    except Exception as e:
        print(f"✗ Config Check: Failed to load config: {e}")
        all_good = False
    
    # Check data file
    print("\n📊 Data:")
    data_path = "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json"
    if check_file_exists(data_path, "Training Data"):
        try:
            import pandas as pd
            df = pd.read_json(data_path)
            print(f"✓ Data Loading: Successfully loaded {len(df)} samples")
            
            # Check essential columns
            required_cols = ['Image', 'set']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"✗ Data Columns: Missing columns: {missing_cols}")
                all_good = False
            else:
                print("✓ Data Columns: All required columns present")
        except Exception as e:
            print(f"✗ Data Loading: Failed to load data: {e}")
            all_good = False
    else:
        all_good = False
    
    # Summary
    print("\n" + "="*70)
    if all_good:
        print("🎉 VERIFICATION PASSED!")
        print("✅ All components are ready for concurrent training")
        print("\n📋 To start training:")
        print("   python train_concurrent_v5.py")
        print("   # or with external test split:")
        print("   python train_concurrent_v5.py --test_split_file path/to/test_ids.txt")
    else:
        print("❌ VERIFICATION FAILED!")
        print("🔧 Please fix the issues above before running concurrent training")
        
        print("\n🛠️  Quick fixes:")
        print("1. Ensure all files are in the correct directory")
        print("2. Install missing dependencies: pip install mmengine mmpose")
        print("3. Check that the data file path is correct")
        print("4. Verify that custom_hooks is properly added to the config")
    
    print("="*70)

if __name__ == "__main__":
    main() 