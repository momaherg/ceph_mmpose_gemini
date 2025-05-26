import numpy as np
import pandas as pd
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.apis import init_model
from mmengine.dataset import pseudo_collate # For collating samples
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
_original_torch_load = torch.load # Store original with a different name

def safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs) # Call the original

torch.load = safe_torch_load

def diagnose_pipeline_consistency(config_path: str,
                                data_root: str,
                                ann_file: str = 'train_data_pure_old_numpy.json',
                                num_samples_check: int = 3):
    """
    Diagnose consistency between training and evaluation data pipelines.
    """
    
    print("="*80)
    print("PIPELINE CONSISTENCY DIAGNOSIS")
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
    
    # Load config
    try:
        cfg = Config.fromfile(config_path)
        print(f"✓ Config loaded from: {config_path}")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return
    
    # Get dataset instances for training and validation
    print("\n1. LOADING DATASETS & DATALOADERS")
    print("-" * 50)
    
    try:
        # Training dataset
        train_dataset_cfg = cfg.train_dataloader.dataset.copy() # Make a copy to modify
        train_dataset_cfg.pop('type', None) # Remove 'type' if it exists
        train_dataset = custom_cephalometric_dataset.CustomCephalometricDataset(**train_dataset_cfg)
        print(f"✓ Training dataset loaded: {len(train_dataset)} samples")
        
        # Validation dataset (using same config but with test_mode=True)
        val_dataset_cfg = cfg.val_dataloader.dataset.copy() # Make a copy to modify
        val_dataset_cfg.pop('type', None) # Remove 'type' if it exists
        val_dataset = custom_cephalometric_dataset.CustomCephalometricDataset(**val_dataset_cfg)
        print(f"✓ Validation dataset loaded: {len(val_dataset)} samples")
        
    except Exception as e:
        print(f"✗ Failed to load datasets: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. PIPELINE OUTPUT COMPARISON
    print("\n2. PIPELINE OUTPUT COMPARISON")
    print("-" * 50)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("✗ Cannot compare pipelines - one or both datasets are empty.")
        return
        
    for i in range(min(num_samples_check, len(train_dataset), len(val_dataset))):
        print(f"\n--- COMPARING SAMPLE {i} ---")
        
        # Get raw data info (before pipeline)
        # Assuming your custom dataset can return this via a method or by direct access
        # For simplicity, we re-load the raw data for comparison here.
        # This requires the ann_file and data_root to be accessible.
        try:
            data_path = osp.join(data_root, ann_file)
            df = pd.read_json(data_path)
            
            current_train_data_info = train_dataset.get_data_info(i)
            sample_id = current_train_data_info.get('img_id')
            
            print(f"  Attempting to find sample for train_dataset index {i}, img_id: {sample_id} (type: {type(sample_id)})")
            if 'patient_id' in df.columns:
                print(f"  DataFrame patient_id column dtype: {df['patient_id'].dtype}")
            else:
                print("  ⚠️ DataFrame does not have 'patient_id' column!")
                continue

            if sample_id is None:
                 print(f"    ⚠️ Could not get sample_id for train_dataset sample {i}")
                 continue
            
            # Ensure consistent type for comparison if patient_id is numeric
            if pd.api.types.is_numeric_dtype(df['patient_id']):
                try:
                    sample_id_numeric = int(sample_id) 
                    raw_sample_df = df[df['patient_id'] == sample_id_numeric]
                except ValueError:
                    print(f"    ⚠️ Could not convert sample_id '{sample_id}' to numeric for matching.")
                    raw_sample_df = pd.DataFrame() # Empty
            else: # Assume string comparison
                raw_sample_df = df[df['patient_id'] == str(sample_id)]
            
            if raw_sample_df.empty:
                print(f"    ⚠️ Raw sample with patient_id '{sample_id}' not found in main DataFrame df.")
                # As a fallback, try to get raw data based on the order if IDs fail.
                # This assumes the initial order in train_dataset.data_list corresponds to df order.
                # This is less robust and should only be a fallback.
                if i < len(df):
                    print(f"    Trying fallback: using df.iloc[{i}] for raw data.")
                    raw_sample_row = df.iloc[i]
                    fallback_patient_id = raw_sample_row.get('patient_id', 'Unknown')
                    print(f"    Fallback patient_id: {fallback_patient_id}")
                else:
                    print(f"    Fallback failed: index {i} is out of bounds for main DataFrame.")
                    continue
            else:
                raw_sample_row = raw_sample_df.iloc[0]
            
            retrieved_patient_id = raw_sample_row.get('patient_id', 'Unknown')
            print(f"  Raw Sample ID (retrieved): {retrieved_patient_id}")
            raw_img_array = raw_sample_row['Image']
            raw_img_np = np.array(raw_img_array, dtype=np.uint8).reshape((224, 224, 3))
            print(f"  Raw Image Shape: {raw_img_np.shape}, Min/Max: {raw_img_np.min()}/{raw_img_np.max()}")
            
            # Get ground truth keypoints from raw data
            raw_gt_keypoints = np.zeros((19, 2), dtype=np.float32)
            for k_idx, kp_name in enumerate(landmark_names_in_order):
                x_col, y_col = original_landmark_cols[k_idx*2], original_landmark_cols[k_idx*2+1]
                if x_col in raw_sample_row and y_col in raw_sample_row and pd.notna(raw_sample_row[x_col]) and pd.notna(raw_sample_row[y_col]):
                    raw_gt_keypoints[k_idx, 0], raw_gt_keypoints[k_idx, 1] = raw_sample_row[x_col], raw_sample_row[y_col]
            print(f"  Raw GT Keypoints (first 3):\n{raw_gt_keypoints[:3]}")

        except Exception as e:
            print(f"  ⚠️ Error loading raw sample data: {e}")
            continue
            
        # Process with training pipeline
        try:
            train_sample_processed = train_dataset[i]
            print(f"\n  Training Pipeline Output:")
            print(f"    Keys: {train_sample_processed.keys()}")
            train_inputs = train_sample_processed['inputs']
            train_data_sample = train_sample_processed['data_samples']
            print(f"    Inputs Shape: {train_inputs.shape}, Min/Max: {train_inputs.min():.3f}/{train_inputs.max():.3f}")
            print(f"    Data Sample GT Keypoints (first 3):\n{train_data_sample.gt_instances.keypoints[0, :3]}")
            if hasattr(train_data_sample, 'heatmaps'):
                print(f"    Heatmaps Shape: {train_data_sample.heatmaps.shape}, Min/Max: {train_data_sample.heatmaps.min():.3f}/{train_data_sample.heatmaps.max():.3f}")
        except Exception as e:
            print(f"  ✗ Error processing with training pipeline: {e}")
            import traceback
            traceback.print_exc()
            continue
            
        # Process with validation pipeline
        try:
            # Find the same sample in validation dataset (assuming IDs match)
            val_idx = -1
            for k_val in range(len(val_dataset)):
                if val_dataset.get_data_info(k_val).get('img_id') == sample_id:
                    val_idx = k_val
                    break
            
            if val_idx == -1:
                print(f"  ⚠️ Sample ID {sample_id} not found in validation dataset for comparison.")
                continue

            val_sample_processed = val_dataset[val_idx]
            print(f"\n  Validation Pipeline Output:")
            print(f"    Keys: {val_sample_processed.keys()}")
            val_inputs = val_sample_processed['inputs']
            val_data_sample = val_sample_processed['data_samples']
            print(f"    Inputs Shape: {val_inputs.shape}, Min/Max: {val_inputs.min():.3f}/{val_inputs.max():.3f}")
            # Validation pipeline might not generate GT keypoints in data_sample if not needed for evaluation
            if hasattr(val_data_sample, 'gt_instances') and hasattr(val_data_sample.gt_instances, 'keypoints'):
                 print(f"    Data Sample GT Keypoints (first 3):\n{val_data_sample.gt_instances.keypoints[0, :3]}")
            else:
                 print("    Data Sample GT Keypoints: Not present (as expected for some eval pipelines)")

        except Exception as e:
            print(f"  ✗ Error processing with validation pipeline: {e}")
            import traceback
            traceback.print_exc()
            continue
            
        # Comparison (simple check for now)
        if train_inputs.shape == val_inputs.shape:
            # Ensure inputs are float before calculating mean difference
            train_inputs_float = train_inputs.float()
            val_inputs_float = val_inputs.float()
            diff = torch.abs(train_inputs_float - val_inputs_float).mean()
            print(f"\n  Input Tensor Difference (Mean Abs): {diff:.6f}")
            if diff > 1e-3: # Allow for some floating point differences due to different aug paths
                print(f"    ⚠️  POTENTIAL ISSUE: Input tensors differ significantly between pipelines!")
        else:
            print(f"    ⚠️  POTENTIAL ISSUE: Input tensor shapes differ! Train: {train_inputs.shape}, Val: {val_inputs.shape}")

    print("="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    config_path = "/content/ceph_mmpose_gemini/configs/hrnetv2/hrnetv2_w18_cephalometric_224x224_FIXED_V2.py"
    data_root = "/content/drive/MyDrive/Lala's Masters/"
    
    diagnose_pipeline_consistency(config_path, data_root, num_samples_check=3) 