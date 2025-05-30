#!/usr/bin/env python3
"""
Evaluate Cephalometric Model on Training Samples
This helps diagnose if the model is overfitting or has fundamental learning issues.
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
import functools
original_torch_load = torch.load

def safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

torch.load = safe_torch_load

def evaluate_on_training_samples(checkpoint_path: str,
                                config_path: str = "/content/ceph_mmpose_gemini/configs/hrnetv2/hrnetv2_w18_cephalometric_224x224_FIXED_V2.py",
                                data_root: str = "/content/drive/MyDrive/Lala's Masters/",
                                ann_file: str = 'train_data_pure_old_numpy.json',
                                num_samples: int = 50,
                                sample_type: str = 'train') -> Dict:
    """
    Evaluate model on training samples to check for overfitting vs learning issues.
    
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
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return {}
    
    # Load dataset
    try:
        data_path = osp.join(data_root, ann_file)
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
    
    # Debug: Check first sample structure
    first_sample = df_sample.iloc[0]
    print(f"DEBUG - Available columns: {list(first_sample.index)}")
    if 'Image' in first_sample:
        print(f"DEBUG - Image type: {type(first_sample['Image'])}")
        if hasattr(first_sample['Image'], 'shape'):
            print(f"DEBUG - Image shape: {first_sample['Image'].shape}")
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
                
                # Prepare input
                if idx == 0:
                    print(f"DEBUG - Preparing tensor input...")
                
                image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
                image_tensor = image_tensor.cuda()
                
                if idx == 0:
                    print(f"DEBUG - Tensor shape: {image_tensor.shape}")
                    print(f"DEBUG - Tensor device: {image_tensor.device}")
                
                # Model inference
                if idx == 0:
                    print(f"DEBUG - Running model inference...")
                
                try:
                    # Create proper MMPose input format
                    from mmpose.structures import PoseDataSample
                    
                    # Create data sample
                    data_sample = PoseDataSample()
                    
                    # Add required metainfo
                    data_sample.set_metainfo({
                        'img_shape': (224, 224),
                        'ori_shape': (224, 224), 
                        'input_size': (224, 224),
                        'input_center': np.array([112.0, 112.0]),
                        'input_scale': np.array([224.0, 224.0]),
                        'flip_indices': list(range(19)), # num_keypoints
                    })
                    
                    # Add gt_instances with bboxes (required by some models)
                    from mmengine.structures import InstanceData
                    gt_instances = InstanceData()
                    gt_instances.bboxes = torch.tensor([[0, 0, 224, 224]], dtype=torch.float32)
                    gt_instances.bbox_scores = torch.tensor([1.0], dtype=torch.float32)
                    data_sample.gt_instances = gt_instances
                    
                    # Create input dict in MMPose format
                    data_dict = {
                        'inputs': image_tensor,
                        'data_samples': [data_sample]
                    }
                    
                    if idx == 0:
                        print(f"DEBUG - Created data_dict with keys: {data_dict.keys()}")
                        print(f"DEBUG - Data samples type: {type(data_dict['data_samples'])}")
                    
                    # Use the model's test_step with proper format
                    results = model.test_step(data_dict)
                    
                    if idx == 0:
                        print(f"DEBUG - Model results type: {type(results)}")
                        print(f"DEBUG - Model results: {results}")
                        if results is not None:
                            print(f"DEBUG - Model results length: {len(results) if results else 'None'}")
                        
                except Exception as e:
                    if idx == 0:
                        print(f"DEBUG - MMPose format failed: {e}")
                        print(f"DEBUG - Trying predict method with data_samples...")
                    
                    try:
                        # Try predict method with proper format
                        from mmpose.structures import PoseDataSample
                        data_sample = PoseDataSample()
                        
                        results = model.predict(image_tensor, [data_sample])
                        
                        if idx == 0:
                            print(f"DEBUG - Predict method succeeded: {type(results)}")
                        
                    except Exception as e2:
                        if idx == 0:
                            print(f"DEBUG - Predict method failed: {e2}")
                            print(f"DEBUG - Trying inference_topdown from mmpose.apis...")
                        
                        try:
                            # Try using MMPose's inference_topdown API (more robust)
                            from mmpose.apis import inference_topdown # Changed from inference_model
                            
                            # Convert tensor back to numpy for inference_topdown
                            image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                            
                            # Denormalize and convert to uint8 if needed by inference_topdown
                            # This depends on whether your inference_topdown expects raw or normalized images
                            # For now, assume it handles normalization if needed or uses raw uint8
                            # mean = np.array([123.675, 116.28, 103.53])
                            # std = np.array([58.395, 57.12, 57.375])
                            # image_np = (image_np * std) + mean
                            image_np = image_np.astype(np.uint8) # Ensure uint8 for some APIs

                            # Bbox for the whole image
                            bbox = np.array([0, 0, 224, 224])
                            
                            results_list = inference_topdown(model, image_np, bboxes=bbox.reshape(1, -1))
                            
                            if idx == 0:
                                print(f"DEBUG - inference_topdown succeeded: {type(results_list)}")
                            # Process results_list to fit the expected 'results' structure if needed
                            # For now, assuming it gives a list of PoseDataSample or similar
                            if results_list and len(results_list) > 0:
                                results = results_list # Assign to results to be processed later
                            else:
                                print("DEBUG - inference_topdown returned empty results")
                                continue
                            
                        except ImportError:
                            if idx == 0:
                                print("DEBUG - inference_topdown not available.")
                            continue # Skip to next sample if this API is not found
                        except Exception as e3:
                            if idx == 0:
                                print(f"DEBUG - All inference methods failed. Last error from inference_topdown: {e3}")
                            continue
                
                # Validate results
                if not results or len(results) == 0:
                    if idx == 0:
                        print(f"DEBUG - No results from model")
                    continue
                
                # Extract predictions
                if idx == 0:
                    print(f"DEBUG - Extracting predictions...")
                    print(f"DEBUG - Results[0] type: {type(results[0])}")
                    print(f"DEBUG - Has pred_instances: {hasattr(results[0], 'pred_instances')}")
                
                # pred_coords = results[0].pred_instances.keypoints[0].cpu().numpy()  # Shape: (19, 2)
                pred_coords = results[0].pred_instances.keypoints
                # Ensure it's a numpy array and has the correct dimensions
                if isinstance(pred_coords, torch.Tensor):
                    pred_coords = pred_coords.cpu().numpy()
                
                if pred_coords.shape == (1, 19, 2): # Remove batch dim if present
                    pred_coords = pred_coords[0]
                elif pred_coords.shape != (19,2):
                    print(f"ERROR: Unexpected pred_coords shape: {pred_coords.shape}")
                    continue
                
                if idx == 0:
                    print(f"DEBUG - Prediction coords shape: {pred_coords.shape}")
                    print(f"DEBUG - First few predictions: {pred_coords[:3]}")
                
                prediction_clusters.append(pred_coords.flatten())
                
                # Extract ground truth
                gt_coords = np.zeros((19, 2))
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
            'model_collapse_detected': True,  # If no predictions, assume collapse
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

# Convenience function for quick evaluation
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
        
        overfitting_ratio = test_results['overall_mre'] / train_results['overall_mre']
        print(f"Test/Train Ratio: {overfitting_ratio:.2f}")
        
        if overfitting_ratio > 1.5:
            print("🔴 OVERFITTING: Test error >> Training error")
        elif overfitting_ratio > 1.2:
            print("🟡 MILD OVERFITTING: Test error > Training error")
        else:
            print("🟢 GOOD GENERALIZATION: Similar train/test performance")
    
    return train_results, test_results

def test_model_inference(checkpoint_path: str,
                        config_path: str = "/content/ceph_mmpose_gemini/configs/hrnetv2/hrnetv2_w18_cephalometric_224x224_FIXED_V2.py"):
    """Simple test of model inference to debug issues."""
    
    print("="*50)
    print("TESTING MODEL INFERENCE")
    print("="*50)
    
    # Initialize scope
    init_default_scope('mmpose')
    
    # Import custom modules
    try:
        import custom_cephalometric_dataset
        import custom_transforms
        import cephalometric_dataset_info
        print("✓ Custom modules imported")
    except ImportError as e:
        print(f"✗ Custom modules import failed: {e}")
        return
    
    # Load model
    try:
        model = init_model(config_path, checkpoint_path, device='cuda:0')
        print(f"✓ Model loaded")
        print(f"  Model type: {type(model)}")
        print(f"  Has test_step: {hasattr(model, 'test_step')}")
        print(f"  Has predict: {hasattr(model, 'predict')}")
        print(f"  Has forward: {hasattr(model, 'forward')}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    print(f"✓ Created dummy input: {dummy_input.shape}")
    
    model.eval()
    with torch.no_grad():
        # Test different inference methods
        print("\n--- Testing test_step with MMPose format ---")
        try:
            from mmpose.structures import PoseDataSample
            data_sample = PoseDataSample()
            data_dict = {
                'inputs': dummy_input,
                'data_samples': [data_sample]
            }
            results = model.test_step(data_dict)
            print(f"✓ test_step succeeded: {type(results)}")
            if results:
                print(f"  Results length: {len(results)}")
                if hasattr(results[0], 'pred_instances'):
                    print(f"  Has pred_instances: True")
                    if hasattr(results[0].pred_instances, 'keypoints'):
                        print(f"  Keypoints shape: {results[0].pred_instances.keypoints.shape}")
        except Exception as e:
            print(f"✗ test_step failed: {e}")
        
        print("\n--- Testing predict method ---")
        try:
            from mmpose.structures import PoseDataSample
            data_sample = PoseDataSample()
            results = model.predict(dummy_input, [data_sample])
            print(f"✓ Predict succeeded: {type(results)}")
        except Exception as e:
            print(f"✗ Predict failed: {e}")
        
        print("\n--- Testing inference_topdown ---")
        try:
            from mmpose.apis import inference_topdown # Changed from inference_model
            # Convert tensor to numpy image
            dummy_np = dummy_input.squeeze(0).permute(1, 2, 0).cpu().numpy()
            dummy_np = (dummy_np * 255).astype(np.uint8)
            results = inference_topdown(model, dummy_np)
            print(f"✓ inference_topdown succeeded: {type(results)}")
        except Exception as e:
            print(f"✗ inference_topdown failed: {e}")
            
        print("\n--- Testing direct forward ---")
        try:
            output = model(dummy_input)
            print(f"✓ Forward succeeded: {type(output)}")
        except Exception as e:
            print(f"✗ Forward failed: {e}")

if __name__ == "__main__":
    # Example usage
    checkpoint_path = "/content/ceph_mmpose_gemini/work_dirs/hrnetv2_w18_cephalometric_experiment_FIXED_V2/epoch_31.pth"
    
    # Test model inference first
    test_model_inference(checkpoint_path)
    
    # Then run full diagnostic
    # train_results, test_results = quick_training_check(checkpoint_path, num_samples=50) 