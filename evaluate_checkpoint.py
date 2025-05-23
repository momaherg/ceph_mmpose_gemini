import numpy as np
import pandas as pd
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.apis import inference_topdown
from mmpose.apis import init_model
from mmpose.registry import DATASETS
import os
import os.path as osp
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# PyTorch 2.6 compatibility for MMEngine checkpoints
try:
    from mmengine.config.config import ConfigDict
    from mmengine.logging.history_buffer import HistoryBuffer
    # Add common MMEngine classes to safe globals for PyTorch 2.6+ compatibility
    mmengine_classes = [ConfigDict, HistoryBuffer]
    
    # Try to import other common MMEngine classes that might be in checkpoints
    try:
        from mmengine.logging.logger import MMLogger
        mmengine_classes.append(MMLogger)
    except ImportError:
        pass
    
    try:
        from mmengine.registry.registry import Registry
        mmengine_classes.append(Registry)
    except ImportError:
        pass
        
    try:
        from mmengine.utils.misc import DefaultScope
        mmengine_classes.append(DefaultScope)
    except ImportError:
        pass
    
    torch.serialization.add_safe_globals(mmengine_classes)
    print(f"Added {len(mmengine_classes)} MMEngine classes to PyTorch safe globals for checkpoint loading")
    print(f"Classes: {[cls.__name__ for cls in mmengine_classes]}")
except (ImportError, AttributeError) as e:
    print(f"Note: Could not add MMEngine classes to safe globals: {e}")
    # Fallback: try to set weights_only=False globally if possible
    pass

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

def calculate_mre(pred_keypoints: np.ndarray, gt_keypoints: np.ndarray, 
                  keypoints_visible: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate Mean Radial Error (MRE) between predicted and ground truth keypoints.
    
    Args:
        pred_keypoints: Predicted keypoints, shape (N, K, 2) where N=samples, K=keypoints
        gt_keypoints: Ground truth keypoints, shape (N, K, 2)
        keypoints_visible: Visibility mask, shape (N, K). If None, all keypoints are considered visible.
    
    Returns:
        Dictionary containing MRE metrics
    """
    if pred_keypoints.shape != gt_keypoints.shape:
        raise ValueError(f"Shape mismatch: pred {pred_keypoints.shape} vs gt {gt_keypoints.shape}")
    
    N, K, _ = pred_keypoints.shape
    
    if keypoints_visible is None:
        keypoints_visible = np.ones((N, K))
    
    # Calculate Euclidean distance for each keypoint
    distances = np.sqrt(np.sum((pred_keypoints - gt_keypoints) ** 2, axis=2))  # Shape: (N, K)
    
    # Apply visibility mask (only consider visible keypoints)
    valid_mask = keypoints_visible > 0
    masked_distances = distances * valid_mask
    
    # Calculate MRE per keypoint (average across samples)
    mre_per_keypoint = []
    for k in range(K):
        valid_samples = valid_mask[:, k].sum()
        if valid_samples > 0:
            mre_k = masked_distances[:, k].sum() / valid_samples
            mre_per_keypoint.append(mre_k)
        else:
            mre_per_keypoint.append(0.0)
    
    # Overall MRE (average across all valid keypoints and samples)
    total_valid = valid_mask.sum()
    overall_mre = masked_distances.sum() / total_valid if total_valid > 0 else 0.0
    
    # Standard deviation of errors
    valid_distances = distances[valid_mask]
    mre_std = np.std(valid_distances) if len(valid_distances) > 0 else 0.0
    
    return {
        'overall_mre': float(overall_mre),
        'mre_std': float(mre_std),
        'mre_per_keypoint': mre_per_keypoint,
        'num_valid_samples': int(total_valid),
        'num_total_samples': int(N * K)
    }

def load_test_dataset(data_root: str, test_ann_file: str) -> pd.DataFrame:
    """
    Load test dataset from JSON file and filter for test set.
    
    Args:
        data_root: Root directory containing the data
        test_ann_file: Annotation file name (relative to data_root) containing both train and test data
    
    Returns:
        Pandas DataFrame with test data only
    """
    test_path = osp.join(data_root, test_ann_file)
    if not osp.exists(test_path):
        raise FileNotFoundError(f"Data file not found: {test_path}")
    
    df = pd.read_json(test_path)
    print(f"Loaded dataset with {len(df)} total samples from {test_path}")
    
    # Filter for test set only
    if 'set' in df.columns:
        test_df = df[df['set'] == 'test'].copy()
        print(f"Filtered to {len(test_df)} test samples")
        
        if len(test_df) == 0:
            print("Warning: No test samples found. Available sets:")
            print(df['set'].value_counts())
            raise ValueError("No test samples found in the dataset")
    else:
        print("Warning: 'set' column not found. Using all data as test set.")
        test_df = df.copy()
    
    return test_df

def preprocess_image(img_array: List, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess image data for model inference.
    
    Args:
        img_array: Image data as list (from JSON)
        target_size: Target image size (height, width)
    
    Returns:
        Preprocessed image as numpy array
    """
    img_np = np.array(img_array, dtype=np.uint8).reshape((*target_size, 3))
    return img_np

def extract_ground_truth_keypoints(row: pd.Series, 
                                 landmark_names: List[str], 
                                 landmark_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract ground truth keypoints and visibility from a data row.
    
    Args:
        row: Data row from DataFrame
        landmark_names: List of landmark names in order
        landmark_cols: List of original landmark column names
    
    Returns:
        Tuple of (keypoints, visibility) arrays
    """
    num_keypoints = len(landmark_names)
    keypoints = np.zeros((num_keypoints, 2), dtype=np.float32)
    keypoints_visible = np.ones(num_keypoints, dtype=np.int32)
    
    for i, kp_name in enumerate(landmark_names):
        x_col = landmark_cols[i*2]
        y_col = landmark_cols[i*2+1]
        
        if x_col in row and y_col in row and pd.notna(row[x_col]) and pd.notna(row[y_col]):
            keypoints[i, 0] = row[x_col]
            keypoints[i, 1] = row[y_col]
        else:
            keypoints[i, 0] = 0
            keypoints[i, 1] = 0
            keypoints_visible[i] = 0
    
    return keypoints, keypoints_visible

def evaluate_checkpoint(checkpoint_path: str,
                       config_path: str,
                       data_root: str,
                       test_ann_file: str = 'train_data_pure_old_numpy.json',
                       device: str = 'cuda:0',
                       batch_size: int = 1) -> Dict:
    """
    Evaluate a trained checkpoint on the test set using MRE metric.
    
    Args:
        checkpoint_path: Path to the saved checkpoint (.pth file)
        config_path: Path to the config file used for training
        data_root: Root directory containing the test data
        test_ann_file: Annotation file name (relative to data_root) - will filter for 'set'=='test'
        device: Device to use for inference ('cuda:0', 'cpu', etc.)
        batch_size: Batch size for inference (keep it small to avoid memory issues)
    
    Returns:
        Dictionary containing evaluation results
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
    except Exception as e:
        error_msg = str(e)
        if "weights_only" in error_msg or "WeightsUnpickler" in error_msg:
            print(f"PyTorch 2.6 checkpoint loading error detected: {e}")
            print("\nTrying alternative loading methods...")
            
            # Try with safe_globals context manager
            try:
                from mmengine.config.config import ConfigDict
                from mmengine.logging.history_buffer import HistoryBuffer
                
                # Collect all MMEngine classes that might be in checkpoints
                safe_classes = [ConfigDict, HistoryBuffer]
                
                # Add other classes if available
                try:
                    from mmengine.logging.logger import MMLogger
                    safe_classes.append(MMLogger)
                except ImportError:
                    pass
                    
                try:
                    from mmengine.registry.registry import Registry
                    safe_classes.append(Registry)
                except ImportError:
                    pass
                    
                try:
                    from mmengine.utils.misc import DefaultScope
                    safe_classes.append(DefaultScope)
                except ImportError:
                    pass
                
                with torch.serialization.safe_globals(safe_classes):
                    model = init_model(config_path, checkpoint_path, device=device)
                print("Model loaded successfully using safe_globals context manager!")
            except Exception as e2:
                print(f"Context manager approach failed: {e2}")
                print("\nTo fix this PyTorch 2.6 compatibility issue, you can:")
                print("1. Downgrade PyTorch: pip install torch==2.5.1")
                print("2. Or add this code before evaluation:")
                print("   import torch")
                print("   from mmengine.config.config import ConfigDict")
                print("   from mmengine.logging.history_buffer import HistoryBuffer")
                print("   torch.serialization.add_safe_globals([ConfigDict, HistoryBuffer])")
                print("3. Or set weights_only=False in PyTorch load calls (less secure)")
                raise RuntimeError("Failed to load checkpoint due to PyTorch 2.6 compatibility issue")
        else:
            print(f"Error loading model: {e}")
            raise
    
    # Load test dataset
    try:
        test_df = load_test_dataset(data_root, test_ann_file)
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        raise
    
    # Get dataset metadata
    if CUSTOM_MODULES_AVAILABLE:
        landmark_names = landmark_names_in_order
        landmark_cols = original_landmark_cols
        num_keypoints = len(landmark_names)
    else:
        # Fallback - assume standard cephalometric landmarks
        num_keypoints = 19
        landmark_names = [f"landmark_{i}" for i in range(num_keypoints)]
        landmark_cols = [f"x{i}" for i in range(num_keypoints)] + [f"y{i}" for i in range(num_keypoints)]
        print("Using fallback landmark names due to missing custom modules")
    
    print(f"Evaluating on {len(test_df)} test samples with {num_keypoints} keypoints")
    
    # Storage for predictions and ground truth
    all_pred_keypoints = []
    all_gt_keypoints = []
    all_gt_visible = []
    
    # Process test samples
    print("Starting inference on test set...")
    for idx, (_, row) in enumerate(test_df.iterrows()):
        if idx % 50 == 0:
            print(f"Processing sample {idx+1}/{len(test_df)}")
        
        try:
            # Preprocess image
            img_np = preprocess_image(row['Image'])
            
            # Extract ground truth
            gt_keypoints, gt_visible = extract_ground_truth_keypoints(
                row, landmark_names, landmark_cols
            )
            
            # Run inference
            # Create a simple bbox for the entire image
            bbox = np.array([0, 0, 224, 224])  # [x1, y1, x2, y2]
            
            try:
                # Run inference using MMPose API
                result = inference_topdown(model, img_np, bboxes=bbox.reshape(1, -1))
                
                if len(result) > 0 and 'keypoints' in result[0]:
                    pred_keypoints = result[0]['keypoints']  # Shape should be (num_keypoints, 2)
                    if pred_keypoints.shape[0] != num_keypoints:
                        print(f"Warning: Expected {num_keypoints} keypoints, got {pred_keypoints.shape[0]}")
                        continue
                else:
                    print(f"Warning: No valid predictions for sample {idx}")
                    continue
                    
            except Exception as e:
                print(f"Inference failed for sample {idx}: {e}")
                continue
            
            # Store results
            all_pred_keypoints.append(pred_keypoints)
            all_gt_keypoints.append(gt_keypoints)
            all_gt_visible.append(gt_visible)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    if not all_pred_keypoints:
        raise RuntimeError("No valid predictions obtained from the test set")
    
    # Convert to numpy arrays
    pred_keypoints_array = np.array(all_pred_keypoints)  # Shape: (N, K, 2)
    gt_keypoints_array = np.array(all_gt_keypoints)      # Shape: (N, K, 2)
    gt_visible_array = np.array(all_gt_visible)          # Shape: (N, K)
    
    print(f"Successfully processed {len(all_pred_keypoints)} samples")
    print(f"Prediction shape: {pred_keypoints_array.shape}")
    print(f"Ground truth shape: {gt_keypoints_array.shape}")
    
    # Calculate MRE
    print("Calculating MRE metrics...")
    mre_results = calculate_mre(pred_keypoints_array, gt_keypoints_array, gt_visible_array)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Overall MRE: {mre_results['overall_mre']:.4f} pixels")
    print(f"MRE Std Dev: {mre_results['mre_std']:.4f} pixels")
    print(f"Valid samples: {mre_results['num_valid_samples']}/{mre_results['num_total_samples']}")
    print(f"Number of test images: {len(all_pred_keypoints)}")
    
    if CUSTOM_MODULES_AVAILABLE and len(mre_results['mre_per_keypoint']) == len(landmark_names):
        print("\nMRE per landmark:")
        for i, (name, mre_val) in enumerate(zip(landmark_names, mre_results['mre_per_keypoint'])):
            print(f"  {i+1:2d}. {name:20s}: {mre_val:.4f} pixels")
    
    # Prepare return dictionary
    evaluation_results = {
        'checkpoint_path': checkpoint_path,
        'config_path': config_path,
        'test_samples': len(all_pred_keypoints),
        'mre_metrics': mre_results,
        'predictions': pred_keypoints_array,
        'ground_truth': gt_keypoints_array,
        'visibility': gt_visible_array
    }
    
    return evaluation_results

# Example usage function for Colab
def evaluate_cephalometric_checkpoint(checkpoint_path: str, 
                                    work_dir: str = '/content/work_dirs',
                                    data_root: str = '/content/drive/MyDrive/Lala\'s Masters/'):
    """
    Convenience function for evaluating cephalometric checkpoints in Colab.
    
    Args:
        checkpoint_path: Path to checkpoint file
        work_dir: Working directory (where configs are stored)
        data_root: Data root directory
    
    Returns:
        Evaluation results dictionary
    """
    
    # Determine config path based on checkpoint path or use default
    if 'hrnetv2' in checkpoint_path.lower():
        config_path = osp.join(work_dir, 'configs/hrnetv2/hrnetv2_w18_cephalometric_224x224.py')
    else:
        # Try to find config in the same directory as checkpoint
        checkpoint_dir = osp.dirname(checkpoint_path)
        possible_configs = [
            osp.join(checkpoint_dir, 'hrnetv2_w18_cephalometric_224x224.py'),
            osp.join(work_dir, 'configs/hrnetv2/hrnetv2_w18_cephalometric_224x224.py'),
        ]
        config_path = None
        for cfg in possible_configs:
            if osp.exists(cfg):
                config_path = cfg
                break
        
        if config_path is None:
            raise FileNotFoundError(f"Could not find config file. Tried: {possible_configs}")
        
    config_path = "/content/ceph_mmpose_gemini/configs/hrnetv2/hrnetv2_w18_cephalometric_224x224.py"
    
    return evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        data_root=data_root,
        test_ann_file='train_data_pure_old_numpy.json',
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
    )

if __name__ == "__main__":
    # Example usage
    # Make sure to update these paths according to your setup
    
    # Example checkpoint path (update this to your actual checkpoint)
    checkpoint_path = "work_dirs/hrnetv2_w18_cephalometric_experiment/best_PCKAccuracy_epoch_45.pth"
    
    # Example config path
    config_path = "/content/ceph_mmpose_gemini/configs/hrnetv2/hrnetv2_w18_cephalometric_224x224.py"
    
    # Data root
    data_root = "/content/drive/MyDrive/Lala's Masters/"
    
    try:
        results = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            data_root=data_root
        )
        print(f"\nEvaluation completed successfully!")
        print(f"Overall MRE: {results['mre_metrics']['overall_mre']:.4f} pixels")
        
    except Exception as e:
        print(f"Evaluation failed: {e}") 