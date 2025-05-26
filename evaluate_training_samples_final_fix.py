#!/usr/bin/env python3
"""
FINAL FIXED Evaluation Script for Cephalometric Model
This version uses direct model forward pass to avoid MMPose API issues.
"""

import numpy as np
import pandas as pd
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.apis import init_model
import os
import os.path as osp
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
original_torch_load = torch.load

def safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

torch.load = safe_torch_load

def decode_heatmaps_to_coords(heatmaps):
    """
    Decode heatmaps to coordinates using argmax.
    This replaces the problematic MMPose API calls.
    
    Args:
        heatmaps: Tensor of shape (batch_size, num_keypoints, height, width)
    
    Returns:
        numpy array of shape (num_keypoints, 2) with (x, y) coordinates
    """
    if len(heatmaps.shape) == 4:
        heatmaps = heatmaps[0]  # Remove batch dimension
    
    num_keypoints, height, width = heatmaps.shape
    coords = np.zeros((num_keypoints, 2))
    
    for i in range(num_keypoints):
        heatmap = heatmaps[i].cpu().numpy()
        
        # Find the maximum value location
        max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        y, x = max_idx
        
        # Scale from heatmap size to image size (224x224)
        scale_x = 224.0 / width
        scale_y = 224.0 / height
        
        coords[i, 0] = x * scale_x  # x coordinate
        coords[i, 1] = y * scale_y  # y coordinate
    
    return coords

def evaluate_on_training_samples(checkpoint_path: str,
                                config_path: str = "configs/hrnetv2/hrnetv2_w18_cephalometric_224x224_FIXED_V2.py",
                                data_root: str = "data/",
                                ann_file: str = 'train_data_pure.pkl',
                                num_samples: int = 50,
                                sample_type: str = 'train') -> Dict:
    """
    Evaluate model on training samples using FIXED inference approach.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        data_root: Root directory of data
        ann_file: Annotation file name
        num_samples: Number of samples to evaluate
        sample_type: 'train', 'test', or 'random' for sampling strategy
    
    Returns:
        Dictionary with evaluation results
    """
    
    print("="*80)
    print(f"EVALUATING ON TRAINING SAMPLES - {sample_type.upper()} SET")
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
        return {}
    
    # Load model
    try:
        model = init_model(config_path, checkpoint_path, device='cuda:0')
        print(f"✓ Model loaded from: {checkpoint_path}")
        print(f"  Model type: {type(model)}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return {}
    
    # Load dataset
    try:
        data_path = osp.join(data_root, ann_file)
        
        # Handle both pickle and JSON files
        if ann_file.endswith('.pkl'):
            import pickle
            with open(data_path, 'rb') as f:
                df = pickle.load(f)
            if not isinstance(df, pd.DataFrame):
                print(f"✗ Pickle file does not contain a DataFrame")
                return {}
        else:
            df = pd.read_json(data_path)
            
        print(f"✓ Dataset loaded: {len(df)} total samples")
        
        # Filter by set type
        if sample_type == 'train' and 'set' in df.columns:
            df_filtered = df[df['set'] == 'train'].copy()
            print(f"✓ Filtered to {len(df_filtered)} training samples")
        elif sample_type == 'test' and 'set' in df.columns:
            df_filtered = df[df['set'] == 'test'].copy()
            print(f"✓ Filtered to {len(df_filtered)} test samples")
        else:
            df_filtered = df.copy()
            print(f"✓ Using all {len(df_filtered)} samples")
        
        # Sample subset
        if len(df_filtered) > num_samples:
            df_sample = df_filtered.sample(n=num_samples, random_state=42)
            print(f"✓ Randomly sampled {num_samples} from {len(df_filtered)} available")
        else:
            df_sample = df_filtered
            print(f"✓ Using all {len(df_sample)} available samples")
            
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return {}
    
    # Prepare landmark columns
    landmark_cols = original_landmark_cols
    x_cols = landmark_cols[::2]  # Every other starting from 0
    y_cols = landmark_cols[1::2]  # Every other starting from 1
    
    # Evaluate samples
    print(f"\nEvaluating {len(df_sample)} samples...")
    print("-" * 50)
    
    all_errors = []
    per_landmark_errors = {name: [] for name in landmark_names_in_order}
    valid_predictions = 0
    model_collapse_detected = False
    prediction_clusters = []
    
    model.eval()
    with torch.no_grad():
        for idx, (_, row) in enumerate(df_sample.iterrows()):
            try:
                if idx == 0:  # Debug first sample in detail
                    print(f"DEBUG - Processing sample {idx}")
                    print(f"DEBUG - Patient ID: {row.get('patient_id', 'N/A')}")
                
                # Load and preprocess image
                if 'Image' in row and row['Image'] is not None:
                    try:
                        # Image is stored as array in the dataset
                        image_array = row['Image']
                        if idx == 0:
                            print(f"DEBUG - Image array type: {type(image_array)}")
                            print(f"DEBUG - Image array length: {len(image_array) if image_array else 'None'}")
                        
                        if isinstance(image_array, (list, np.ndarray)):
                            image = np.array(image_array)
                            if idx == 0:
                                print(f"DEBUG - Converted image shape: {image.shape}")
                            
                            # Reshape from (50176, 3) to (224, 224, 3) if needed
                            if image.shape == (50176, 3):
                                image = image.reshape(224, 224, 3)
                                if idx == 0:
                                    print(f"DEBUG - Reshaped to: {image.shape}")
                            elif image.shape != (224, 224, 3):
                                print(f"⚠️  Invalid image shape: {image.shape}")
                                continue
                        else:
                            print(f"⚠️  Invalid image data type: {type(image_array)}")
                            continue
                    except Exception as e:
                        print(f"⚠️  Error processing image array: {e}")
                        continue
                else:
                    # Fallback: try loading from .npy file
                    image_path = osp.join(data_root, f"{row['patient_id']}.npy")
                    if not osp.exists(image_path):
                        print(f"⚠️  Image not found in dataset or file: {image_path}")
                        continue
                    
                    image = np.load(image_path)
                    if image.shape != (224, 224, 3):
                        print(f"⚠️  Invalid image shape: {image.shape}")
                        continue
                
                # Normalize image to [0, 1] range
                if image.max() > 1.0:
                    image = image.astype(np.float32) / 255.0
                
                # Prepare input tensor
                if idx == 0:
                    print(f"DEBUG - Preparing tensor input...")
                
                image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
                image_tensor = image_tensor.cuda()
                
                if idx == 0:
                    print(f"DEBUG - Tensor shape: {image_tensor.shape}")
                    print(f"DEBUG - Tensor device: {image_tensor.device}")
                    print(f"DEBUG - Tensor range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
                
                # FIXED MODEL INFERENCE - No more MMPose API issues!
                if idx == 0:
                    print(f"DEBUG - Running FIXED model inference...")
                
                try:
                    # Use direct forward pass - this is the key fix!
                    outputs = model(image_tensor)
                    
                    if idx == 0:
                        print(f"DEBUG - Direct forward succeeded")
                        print(f"DEBUG - Output type: {type(outputs)}")
                        if isinstance(outputs, (list, tuple)):
                            print(f"DEBUG - Output length: {len(outputs)}")
                            if len(outputs) > 0:
                                print(f"DEBUG - First output shape: {outputs[0].shape}")
                        elif hasattr(outputs, 'shape'):
                            print(f"DEBUG - Output shape: {outputs.shape}")
                    
                    # Extract heatmaps
                    if isinstance(outputs, (list, tuple)):
                        heatmaps = outputs[0]  # Usually the first output is heatmaps
                    else:
                        heatmaps = outputs
                    
                    if idx == 0:
                        print(f"DEBUG - Heatmaps shape: {heatmaps.shape}")
                    
                    # Decode heatmaps to coordinates using our fixed function
                    pred_coords = decode_heatmaps_to_coords(heatmaps)
                    
                    if idx == 0:
                        print(f"DEBUG - Decoded coords shape: {pred_coords.shape}")
                        print(f"DEBUG - First few predictions: {pred_coords[:3]}")

                except Exception as e:
                    if idx == 0:
                        print(f"DEBUG - Direct forward failed: {e}")
                    continue  # Skip this sample
                
                # Validate predictions
                if pred_coords is None or pred_coords.shape[0] != 19:
                    if idx == 0:
                        print(f"DEBUG - Invalid prediction shape or None")
                    continue
                
                prediction_clusters.append(pred_coords.flatten())
                
                # Extract ground truth and calculate errors
                sample_errors = []
                
                for i, (x_col, y_col) in enumerate(zip(x_cols, y_cols)):
                    if x_col in row and y_col in row and pd.notna(row[x_col]) and pd.notna(row[y_col]):
                        gt_x, gt_y = float(row[x_col]), float(row[y_col])
                        pred_x, pred_y = pred_coords[i, 0], pred_coords[i, 1]
                        
                        # Skip invalid coordinates
                        if gt_x <= 0 or gt_y <= 0:
                            continue
                        
                        # Calculate error
                        error = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
                        sample_errors.append(error)
                        per_landmark_errors[landmark_names_in_order[i]].append(error)
                        
                        # Check for reasonable predictions (should be in image bounds)
                        if not (0 <= pred_x <= 224 and 0 <= pred_y <= 224):
                            if idx < 5:  # Only show for first few samples
                                print(f"⚠️  Out-of-bounds prediction: ({pred_x:.1f}, {pred_y:.1f})")
                
                if sample_errors:
                    avg_error = np.mean(sample_errors)
                    all_errors.extend(sample_errors)
                    valid_predictions += 1
                    
                    if idx < 5 or avg_error > 50:  # Show first 5 and any with high error
                        print(f"Sample {idx+1:2d}: Avg Error = {avg_error:6.2f} pixels, ID = {row['patient_id']}")
                
            except Exception as e:
                print(f"⚠️  Failed to process sample {idx}: {e}")
                continue
    
    if not all_errors:
        print("✗ No valid predictions found!")
        return {
            'overall_mre': 0.0,
            'overall_std': 0.0,
            'median_error': 0.0,
            'valid_predictions': 0,
            'total_samples': len(df_sample),
            'per_landmark_errors': {name: None for name in landmark_names_in_order},
            'model_collapse_detected': True,
            'sample_type': sample_type,
            'all_errors': [],
            'error': 'No valid predictions found'
        }
    
    # Analyze results
    overall_mre = np.mean(all_errors)
    overall_std = np.std(all_errors)
    median_error = np.median(all_errors)
    
    print(f"\nRESULTS SUMMARY:")
    print("=" * 50)
    print(f"Valid Predictions: {valid_predictions}/{len(df_sample)}")
    print(f"Overall MRE: {overall_mre:.2f} ± {overall_std:.2f} pixels")
    print(f"Median Error: {median_error:.2f} pixels")
    print(f"Min Error: {np.min(all_errors):.2f} pixels")
    print(f"Max Error: {np.max(all_errors):.2f} pixels")
    
    # Check for model collapse
    if len(prediction_clusters) > 5:
        prediction_array = np.array(prediction_clusters)
        pred_std = np.std(prediction_array, axis=0)
        low_variance_coords = np.sum(pred_std < 5.0)  # Coordinates with < 5 pixel variance
        
        if low_variance_coords > len(pred_std) * 0.7:  # If >70% coordinates have low variance
            model_collapse_detected = True
            print(f"\n🚨 MODEL COLLAPSE DETECTED!")
            print(f"   {low_variance_coords}/{len(pred_std)} coordinates have <5px variance")
            print(f"   Predictions are clustered together")
        else:
            print(f"\n✓ No model collapse detected")
            print(f"   {low_variance_coords}/{len(pred_std)} coordinates have <5px variance")
    
    # Per-landmark analysis
    print(f"\nPER-LANDMARK ERRORS:")
    print("-" * 50)
    landmark_results = {}
    for name in landmark_names_in_order:
        if per_landmark_errors[name]:
            errors = per_landmark_errors[name]
            mean_err = np.mean(errors)
            landmark_results[name] = mean_err
            print(f"{name:20s}: {mean_err:6.2f} ± {np.std(errors):5.2f} px ({len(errors):3d} samples)")
        else:
            landmark_results[name] = None
            print(f"{name:20s}: No valid samples")
    
    # Performance interpretation
    print(f"\nPERFORMANCE INTERPRETATION:")
    print("=" * 50)
    
    if model_collapse_detected:
        print("🚨 ISSUE: Model collapse - all predictions are similar")
        print("   → Model is not learning meaningful patterns")
        print("   → Training parameters need adjustment")
    elif overall_mre > 50:
        print("🔴 POOR: Very high error rate")
        print("   → Model may not be learning properly")
        print("   → Check learning rate, architecture, or data")
    elif overall_mre > 20:
        print("🟡 MODERATE: Moderate error rate")
        if sample_type == 'train':
            print("   → Model is learning but not overfitting training data")
            print("   → May need more training epochs or lower learning rate")
        else:
            print("   → Reasonable performance, may need fine-tuning")
    else:
        print("🟢 GOOD: Low error rate")
        if sample_type == 'train':
            print("   → Model is fitting training data well")
            print("   → Check test performance to assess overfitting")
        else:
            print("   → Model is generalizing well")
    
    # Return results
    return {
        'overall_mre': overall_mre,
        'overall_std': overall_std,
        'median_error': median_error,
        'valid_predictions': valid_predictions,
        'total_samples': len(df_sample),
        'per_landmark_errors': landmark_results,
        'model_collapse_detected': model_collapse_detected,
        'sample_type': sample_type,
        'all_errors': all_errors
    }

def quick_training_check(checkpoint_path: str, num_samples: int = 30):
    """Quick check on both training and test samples."""
    
    print("🔍 QUICK TRAINING DIAGNOSTIC")
    print("=" * 60)
    
    # Test on training samples
    print("\n1. TRAINING SAMPLES:")
    train_results = evaluate_on_training_samples(
        checkpoint_path=checkpoint_path,
        num_samples=num_samples,
        sample_type='train'
    )
    
    # Test on test samples  
    print("\n2. TEST SAMPLES:")
    test_results = evaluate_on_training_samples(
        checkpoint_path=checkpoint_path,
        num_samples=num_samples,
        sample_type='test'
    )
    
    # Compare results
    if train_results and test_results:
        print(f"\n📊 COMPARISON:")
        print("=" * 40)
        print(f"Training MRE: {train_results['overall_mre']:.2f} pixels")
        print(f"Test MRE:     {test_results['overall_mre']:.2f} pixels")
        
        if train_results['overall_mre'] > 0:
            overfitting_ratio = test_results['overall_mre'] / train_results['overall_mre']
            print(f"Test/Train Ratio: {overfitting_ratio:.2f}")
            
            if overfitting_ratio > 1.5:
                print("🔴 OVERFITTING: Test error >> Training error")
            elif overfitting_ratio > 1.2:
                print("🟡 MILD OVERFITTING: Test error > Training error")
            else:
                print("🟢 GOOD GENERALIZATION: Similar train/test performance")
        else:
            print("⚠️  Cannot calculate ratio - training error is 0")
    
    return train_results, test_results

if __name__ == "__main__":
    print("🎯 FIXED EVALUATION SCRIPT")
    print("This version uses direct model forward pass to avoid MMPose API issues.")
    print("\nTo use this script:")
    print("1. Ensure you have a trained checkpoint file")
    print("2. Update the checkpoint_path below")
    print("3. Run the evaluation")
    print("\nExample usage:")
    
    # Example - update this path to your actual checkpoint
    checkpoint_path = "work_dirs/hrnetv2_w18_cephalometric_experiment/epoch_60.pth"
    
    if os.path.exists(checkpoint_path):
        print(f"\nRunning evaluation with: {checkpoint_path}")
        train_results, test_results = quick_training_check(checkpoint_path, num_samples=30)
    else:
        print(f"\n⚠️  Checkpoint not found: {checkpoint_path}")
        print("Please update the checkpoint_path variable to point to your trained model.") 