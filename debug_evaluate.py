import numpy as np
import pandas as pd
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.apis import init_model
from mmpose.registry import DATASETS
import os
import os.path as osp
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Fix for PyTorch 2.6+ weights_only issue with MMEngine checkpoints
try:
    from mmengine.config.config import ConfigDict
    torch.serialization.add_safe_globals([ConfigDict])
    print("Added ConfigDict to PyTorch safe globals")
except ImportError:
    print("ConfigDict not found, trying alternative approach")
    pass

# Fallback: temporarily set weights_only=False for torch.load if needed
import functools
original_torch_load = torch.load

def safe_torch_load(*args, **kwargs):
    """Wrapper for torch.load that handles weights_only parameter safely"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

# Apply the wrapper
torch.load = safe_torch_load

# Import your custom modules (make sure these are available in your environment)
try:
    import custom_cephalometric_dataset
    import custom_transforms
    import cephalometric_dataset_info
    from cephalometric_dataset_info import dataset_info, landmark_names_in_order, original_landmark_cols
    CUSTOM_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Custom modules not available: {e}")
    CUSTOM_MODULES_AVAILABLE = False

def debug_evaluate_checkpoint(checkpoint_path: str,
                            config_path: str,
                            data_root: str,
                            test_ann_file: str = 'train_data_pure_old_numpy.json',
                            device: str = 'cuda:0',
                            num_samples_debug: int = 5) -> Dict:
    """
    Debug version of evaluate_checkpoint that provides detailed information about the evaluation process.
    """
    
    # Initialize MMPose scope
    init_default_scope('mmpose')
    
    if not CUSTOM_MODULES_AVAILABLE:
        print("Warning: Custom modules not available. Make sure to import them before calling this function.")
    
    # Check if files exist
    if not osp.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not osp.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Loading model from checkpoint: {checkpoint_path}")
    print(f"Using config: {config_path}")
    print(f"Device: {device}")
    
    # Load model
    try:
        model = init_model(config_path, checkpoint_path, device=device)
        print("Model loaded successfully!")
        print(f"Model in training mode: {model.training}")
        print(f"Model test_cfg: {getattr(model, 'test_cfg', 'None')}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    # Load test dataset
    try:
        test_path = osp.join(data_root, test_ann_file)
        if not osp.exists(test_path):
            raise FileNotFoundError(f"Data file not found: {test_path}")
        
        df = pd.read_json(test_path)
        print(f"Loaded dataset with {len(df)} total samples from {test_path}")
        
        # Filter for test set only
        if 'set' in df.columns:
            test_df = df[df['set'] == 'test'].copy()
            print(f"Filtered to {len(test_df)} test samples")
        else:
            print("Warning: 'set' column not found. Using all data as test set.")
            test_df = df.copy()
            
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        raise
    
    # Get dataset metadata
    if CUSTOM_MODULES_AVAILABLE:
        landmark_names = landmark_names_in_order
        landmark_cols = original_landmark_cols
        num_keypoints = len(landmark_names)
    else:
        num_keypoints = 19
        landmark_names = [f"landmark_{i}" for i in range(num_keypoints)]
        landmark_cols = [f"x{i}" for i in range(num_keypoints)] + [f"y{i}" for i in range(num_keypoints)]
        print("Using fallback landmark names due to missing custom modules")
    
    print(f"\nDebugging first {num_samples_debug} test samples...")
    print("="*80)
    
    # Debug the first few samples
    for debug_idx in range(min(num_samples_debug, len(test_df))):
        row = test_df.iloc[debug_idx]
        print(f"\n--- DEBUG SAMPLE {debug_idx+1} ---")
        print(f"Patient ID: {row.get('patient_id', 'Unknown')}")
        
        # Check image data
        img_array = row['Image']
        img_np = np.array(img_array, dtype=np.uint8).reshape((224, 224, 3))
        print(f"Image shape: {img_np.shape}, dtype: {img_np.dtype}")
        print(f"Image min/max: {img_np.min()}/{img_np.max()}")
        
        # Check ground truth keypoints
        print(f"\nGround Truth Keypoints:")
        gt_keypoints = np.zeros((num_keypoints, 2), dtype=np.float32)
        gt_visible = np.ones(num_keypoints, dtype=np.int32)
        
        for i, kp_name in enumerate(landmark_names):
            x_col = landmark_cols[i*2]
            y_col = landmark_cols[i*2+1]
            
            if x_col in row and y_col in row and pd.notna(row[x_col]) and pd.notna(row[y_col]):
                gt_keypoints[i, 0] = row[x_col]
                gt_keypoints[i, 1] = row[y_col]
                print(f"  {i:2d}. {kp_name:20s}: ({gt_keypoints[i, 0]:6.2f}, {gt_keypoints[i, 1]:6.2f})")
            else:
                gt_keypoints[i, 0] = 0
                gt_keypoints[i, 1] = 0
                gt_visible[i] = 0
                print(f"  {i:2d}. {kp_name:20s}: MISSING/INVALID")
        
        # Run inference
        try:
            # Prepare image for model
            img_for_model = img_np.copy()
            
            # Convert to CHW format if needed
            if len(img_for_model.shape) == 3 and img_for_model.shape[-1] == 3:
                img_for_model = img_for_model.transpose(2, 0, 1)  # HWC to CHW
            
            # Convert to float and normalize
            img_for_model = img_for_model.astype(np.float32)
            
            # Apply normalization (same as in config)
            mean = np.array([123.675, 116.28, 103.53])
            std = np.array([58.395, 57.12, 57.375])
            for i in range(3):
                img_for_model[i] = (img_for_model[i] - mean[i]) / std[i]
            
            print(f"Normalized image min/max: {img_for_model.min():.3f}/{img_for_model.max():.3f}")
            
            # Convert to tensor and add batch dimension
            img_tensor = torch.from_numpy(img_for_model).unsqueeze(0).to(device)
            
            # Create proper data samples for MMPose
            from mmengine.structures import InstanceData
            from mmpose.structures import PoseDataSample
            
            data_sample = PoseDataSample()
            data_sample.set_metainfo({
                'img_shape': (224, 224),
                'ori_shape': (224, 224), 
                'input_size': (224, 224),
                'input_center': np.array([112.0, 112.0]),
                'input_scale': np.array([224.0, 224.0]),
                'flip_indices': list(range(num_keypoints)),
            })
            
            # Add gt_instances
            gt_instances = InstanceData()
            gt_instances.bboxes = torch.tensor([[0, 0, 224, 224]], dtype=torch.float32)
            gt_instances.bbox_scores = torch.tensor([1.0], dtype=torch.float32)
            data_sample.gt_instances = gt_instances
            
            # Prepare batch data
            batch_data = {
                'inputs': img_tensor,
                'data_samples': [data_sample]
            }
            
            # Run inference
            model.eval()
            with torch.no_grad():
                # Temporarily disable flip testing
                original_test_cfg = getattr(model, 'test_cfg', None)
                if original_test_cfg and hasattr(original_test_cfg, 'flip_test'):
                    model.test_cfg = model.test_cfg.copy() if hasattr(model.test_cfg, 'copy') else dict(model.test_cfg)
                    model.test_cfg['flip_test'] = False
                
                outputs = model.test_step(batch_data)
                
                # Restore original test_cfg
                if original_test_cfg:
                    model.test_cfg = original_test_cfg
            
            # Extract predictions
            if isinstance(outputs, list) and len(outputs) > 0:
                output = outputs[0]
                if hasattr(output, 'pred_instances') and hasattr(output.pred_instances, 'keypoints'):
                    pred_keypoints = output.pred_instances.keypoints
                    if isinstance(pred_keypoints, torch.Tensor):
                        pred_keypoints = pred_keypoints.cpu().numpy()
                    elif isinstance(pred_keypoints, np.ndarray):
                        pred_keypoints = pred_keypoints
                    
                    if len(pred_keypoints.shape) == 3 and pred_keypoints.shape[0] == 1:
                        pred_keypoints = pred_keypoints[0]
                    
                    print(f"\nPredicted Keypoints:")
                    print(f"Prediction shape: {pred_keypoints.shape}")
                    
                    # Calculate errors for this sample
                    errors = []
                    for i, kp_name in enumerate(landmark_names):
                        if gt_visible[i] > 0:
                            error = np.sqrt((pred_keypoints[i, 0] - gt_keypoints[i, 0])**2 + 
                                          (pred_keypoints[i, 1] - gt_keypoints[i, 1])**2)
                            errors.append(error)
                            print(f"  {i:2d}. {kp_name:20s}: pred=({pred_keypoints[i, 0]:6.2f}, {pred_keypoints[i, 1]:6.2f}) "
                                  f"gt=({gt_keypoints[i, 0]:6.2f}, {gt_keypoints[i, 1]:6.2f}) error={error:6.2f}")
                        else:
                            print(f"  {i:2d}. {kp_name:20s}: SKIPPED (invalid GT)")
                    
                    avg_error = np.mean(errors) if errors else 0
                    print(f"\nAverage error for this sample: {avg_error:.2f} pixels")
                    
                else:
                    print("ERROR: No keypoints found in model output")
                    print(f"Output type: {type(output)}")
                    if hasattr(output, 'pred_instances'):
                        print(f"Pred instances attributes: {dir(output.pred_instances)}")
            else:
                print("ERROR: No valid outputs from model")
                print(f"Outputs type: {type(outputs)}")
                
        except Exception as e:
            print(f"ERROR during inference: {e}")
            import traceback
            traceback.print_exc()
        
        print("="*80)
    
    return {"debug_complete": True}

# Convenience function for Colab
def debug_cephalometric_checkpoint(checkpoint_path: str, 
                                 work_dir: str = '/content/ceph_mmpose_gemini',
                                 data_root: str = '/content/drive/MyDrive/Lala\'s Masters/',
                                 num_samples: int = 5):
    """Debug a checkpoint with detailed output"""
    
    config_path = osp.join(work_dir, 'configs/hrnetv2/hrnetv2_w18_cephalometric_224x224.py')
    
    return debug_evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        data_root=data_root,
        num_samples_debug=num_samples
    )

if __name__ == "__main__":
    # Example usage for debugging
    checkpoint_path = "/content/ceph_mmpose_gemini/work_dirs/hrnetv2_w18_cephalometric_experiment/epoch_2.pth"
    debug_cephalometric_checkpoint(checkpoint_path, num_samples=3) 