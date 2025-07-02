#!/usr/bin/env python3
"""
Concurrent Joint MLP Performance Evaluation Script
This script evaluates the performance improvement from joint MLP refinement
trained concurrently with HRNetV2. Supports mid-training evaluation.
"""

import os
import sys
import torch
import torch.nn as nn
import warnings
import pandas as pd
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.apis import init_model, inference_topdown
import glob
import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

# Add current directory to path for custom modules
sys.path.insert(0, os.getcwd())

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

class JointMLPRefinementModel(nn.Module):
    """Joint MLP model for landmark coordinate refinement - same as in hook."""
    def __init__(self, input_dim=38, hidden_dim=500, output_dim=38):
        super(JointMLPRefinementModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Main network with residual connection
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Residual projection (if dimensions don't match)
        self.residual_proj = None
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Main forward pass
        out = self.net(x)
        
        # Add residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(x)
        else:
            residual = x
            
        return out + 0.1 * residual  # Small residual weight to start

def apply_joint_mlp_refinement(predictions, mlp_joint, scaler_input, scaler_target, device):
    """Apply joint MLP refinement to predictions."""
    try:
        # Flatten predictions to 38-D vector [x1, y1, x2, y2, ..., x19, y19]
        pred_flat = predictions.flatten().reshape(1, -1)
        
        # Normalize input predictions
        pred_scaled = scaler_input.transform(pred_flat)
        
        # Convert to tensor
        pred_tensor = torch.FloatTensor(pred_scaled).to(device)
        
        # Apply joint MLP refinement
        with torch.no_grad():
            refined_scaled = mlp_joint(pred_tensor).cpu().numpy()
        
        # Denormalize outputs
        refined_flat = scaler_target.inverse_transform(refined_scaled).flatten()
        
        # Reshape back to [19, 2] format
        refined_coords = refined_flat.reshape(19, 2)
        
        return refined_coords
        
    except Exception as e:
        print(f"Joint MLP refinement failed: {e}")
        return predictions

def compute_metrics(pred_coords, gt_coords, landmark_names):
    """Compute comprehensive evaluation metrics."""
    # Compute radial errors
    radial_errors = np.sqrt(np.sum((pred_coords - gt_coords)**2, axis=2))
    
    # Overall metrics
    valid_mask = (gt_coords[:, :, 0] > 0) & (gt_coords[:, :, 1] > 0)
    valid_errors = radial_errors[valid_mask]
    
    overall_metrics = {
        'mre': np.mean(valid_errors),
        'std': np.std(valid_errors),
        'median': np.median(valid_errors),
        'p90': np.percentile(valid_errors, 90),
        'p95': np.percentile(valid_errors, 95),
        'max': np.max(valid_errors)
    }
    
    # Per-landmark metrics
    per_landmark_metrics = {}
    for i, name in enumerate(landmark_names):
        landmark_errors = radial_errors[:, i]
        landmark_valid = valid_mask[:, i]
        
        if np.any(landmark_valid):
            valid_landmark_errors = landmark_errors[landmark_valid]
            per_landmark_metrics[name] = {
                'mre': np.mean(valid_landmark_errors),
                'std': np.std(valid_landmark_errors),
                'median': np.median(valid_landmark_errors),
                'count': len(valid_landmark_errors)
            }
        else:
            per_landmark_metrics[name] = {'mre': 0, 'std': 0, 'median': 0, 'count': 0}
    
    return overall_metrics, per_landmark_metrics

def find_best_available_checkpoint(work_dir):
    """Find the best available checkpoint for evaluation."""
    print(f"🔍 Searching for checkpoints in: {work_dir}")
    
    # Priority order for checkpoint selection
    checkpoint_patterns = [
        "best_NME_epoch_*.pth",  # Best validation performance
        "latest.pth",            # Latest checkpoint
        "epoch_*.pth"            # Any epoch checkpoint
    ]
    
    for pattern in checkpoint_patterns:
        checkpoint_pattern = os.path.join(work_dir, pattern)
        checkpoints = glob.glob(checkpoint_pattern)
        
        if checkpoints:
            if pattern == "epoch_*.pth":
                # Get the latest epoch checkpoint
                checkpoint = max(checkpoints, key=lambda x: int(x.split("epoch_")[1].split(".")[0]))
            else:
                checkpoint = max(checkpoints, key=os.path.getctime)
            
            checkpoint_name = os.path.basename(checkpoint)
            print(f"✓ Found checkpoint: {checkpoint_name}")
            return checkpoint, checkpoint_name
    
    return None, None

def find_synchronized_mlp_model(work_dir, hrnet_checkpoint_name):
    """Find the best synchronized MLP model for the given HRNet checkpoint."""
    mlp_dir = os.path.join(work_dir, "concurrent_mlp")
    mapping_file = os.path.join(mlp_dir, "checkpoint_mlp_mapping.json")
    
    print(f"🔍 Looking for MLP model synchronized with: {hrnet_checkpoint_name}")
    
    # Try synchronized mapping first
    if os.path.exists(mapping_file):
        try:
            import json
            with open(mapping_file, 'r') as f:
                checkpoint_mapping = json.load(f)
            
            if hrnet_checkpoint_name in checkpoint_mapping:
                synchronized_mlp_path = checkpoint_mapping[hrnet_checkpoint_name]
                if os.path.exists(synchronized_mlp_path):
                    print(f"✓ Found synchronized MLP: {os.path.basename(synchronized_mlp_path)}")
                    return synchronized_mlp_path, f"synchronized_with_{hrnet_checkpoint_name}"
                else:
                    print(f"⚠️  Mapped MLP model not found: {synchronized_mlp_path}")
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint mapping: {e}")
    else:
        print(f"💡 No checkpoint mapping found - using fallback strategy")
    
    # Fallback strategies
    print("🔄 Using fallback MLP model selection...")
    
    # Extract epoch number if available
    hrnet_epoch = None
    if "epoch_" in hrnet_checkpoint_name:
        try:
            hrnet_epoch = int(hrnet_checkpoint_name.split("epoch_")[1].split(".")[0])
            print(f"💡 HRNet checkpoint is from epoch {hrnet_epoch}")
        except:
            pass
    
    # Try epoch-specific model
    if hrnet_epoch is not None:
        epoch_mlp_path = os.path.join(mlp_dir, f"mlp_joint_epoch_{hrnet_epoch}.pth")
        if os.path.exists(epoch_mlp_path):
            print(f"✓ Found matching epoch MLP: mlp_joint_epoch_{hrnet_epoch}.pth")
            return epoch_mlp_path, f"epoch_matched_{hrnet_epoch}"
    
    # Try latest epoch model
    epoch_models = glob.glob(os.path.join(mlp_dir, "mlp_joint_epoch_*.pth"))
    if epoch_models:
        latest_epoch_model = max(epoch_models, key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
        epoch_num = latest_epoch_model.split('_epoch_')[1].split('.')[0]
        print(f"✓ Found latest epoch MLP: mlp_joint_epoch_{epoch_num}.pth")
        
        if hrnet_epoch is not None and abs(int(epoch_num) - hrnet_epoch) > 2:
            print(f"⚠️  Warning: MLP epoch {epoch_num} vs HRNet epoch {hrnet_epoch} (mismatch)")
        
        return latest_epoch_model, f"epoch_{epoch_num}_fallback"
    
    # Try latest model
    mlp_latest_path = os.path.join(mlp_dir, "mlp_joint_latest.pth")
    if os.path.exists(mlp_latest_path):
        print(f"✓ Found latest MLP: mlp_joint_latest.pth")
        return mlp_latest_path, "latest_fallback"
    
    # Try final model
    mlp_final_path = os.path.join(mlp_dir, "mlp_joint_final.pth")
    if os.path.exists(mlp_final_path):
        print(f"✓ Found final MLP: mlp_joint_final.pth")
        return mlp_final_path, "final_fallback"
    
    return None, None

def quick_evaluation(work_dir, test_df, landmark_names, landmark_cols, verbose=True):
    """Perform quick evaluation for mid-training monitoring."""
    if verbose:
        print(f"\n🚀 Quick evaluation for: {os.path.basename(work_dir)}")
    
    # Find checkpoints
    hrnet_checkpoint, hrnet_checkpoint_name = find_best_available_checkpoint(work_dir)
    if hrnet_checkpoint is None:
        if verbose:
            print("❌ No HRNet checkpoint found")
        return None
    
    # Find MLP model
    synchronized_mlp_path, model_type = find_synchronized_mlp_model(work_dir, hrnet_checkpoint_name)
    if synchronized_mlp_path is None:
        if verbose:
            print("❌ No MLP model found")
        return None
    
    # Load models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config_path = "Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py"
    
    try:
        # Load HRNet
        hrnet_model = init_model(config_path, hrnet_checkpoint, device=device)
        
        # Load MLP
        mlp_joint = JointMLPRefinementModel().to(device)
        mlp_joint.load_state_dict(torch.load(synchronized_mlp_path, map_location=device))
        mlp_joint.eval()
        
        # Load scalers
        scaler_dir = os.path.join(work_dir, "concurrent_mlp")
        scaler_input = joblib.load(os.path.join(scaler_dir, "scaler_joint_input.pkl"))
        scaler_target = joblib.load(os.path.join(scaler_dir, "scaler_joint_target.pkl"))
        
        if verbose:
            print(f"✓ Models loaded ({model_type})")
        
    except Exception as e:
        if verbose:
            print(f"❌ Failed to load models: {e}")
        return None
    
    # Quick evaluation on subset of test data (for speed)
    eval_subset = min(50, len(test_df))  # Use first 50 samples for quick eval
    subset_df = test_df.head(eval_subset)
    
    hrnet_predictions = []
    mlp_predictions = []
    ground_truths = []
    
    for idx, row in subset_df.iterrows():
        try:
            # Get image and ground truth
            img_array = np.array(row['Image'], dtype=np.uint8).reshape((224, 224, 3))
            
            gt_keypoints = []
            valid_gt = True
            for i in range(0, len(landmark_cols), 2):
                x_col = landmark_cols[i]
                y_col = landmark_cols[i+1]
                if x_col in row and y_col in row and pd.notna(row[x_col]) and pd.notna(row[y_col]):
                    gt_keypoints.append([row[x_col], row[y_col]])
                else:
                    gt_keypoints.append([0, 0])
                    valid_gt = False
            
            if not valid_gt:
                continue
                
            gt_keypoints = np.array(gt_keypoints)
            
            # Run HRNet inference
            bbox = np.array([[0, 0, 224, 224]], dtype=np.float32)
            results = inference_topdown(hrnet_model, img_array, bboxes=bbox, bbox_format='xyxy')
            
            if results and len(results) > 0:
                pred_keypoints = results[0].pred_instances.keypoints[0]
                if isinstance(pred_keypoints, torch.Tensor):
                    pred_keypoints = pred_keypoints.cpu().numpy()
            else:
                continue

            if pred_keypoints is None or pred_keypoints.shape[0] != 19:
                continue
            
            # Apply MLP refinement
            refined_keypoints = apply_joint_mlp_refinement(
                pred_keypoints, mlp_joint, scaler_input, scaler_target, device
            )
            
            hrnet_predictions.append(pred_keypoints)
            mlp_predictions.append(refined_keypoints)
            ground_truths.append(gt_keypoints)
            
        except Exception as e:
            continue
    
    if len(hrnet_predictions) == 0:
        if verbose:
            print("❌ No valid predictions generated")
        return None
    
    # Compute metrics
    hrnet_coords = np.array(hrnet_predictions)
    mlp_coords = np.array(mlp_predictions)
    gt_coords = np.array(ground_truths)
    
    hrnet_overall, hrnet_per_landmark = compute_metrics(hrnet_coords, gt_coords, landmark_names)
    mlp_overall, mlp_per_landmark = compute_metrics(mlp_coords, gt_coords, landmark_names)
    
    # Calculate improvements
    improvement_mre = (hrnet_overall['mre'] - mlp_overall['mre']) / hrnet_overall['mre'] * 100
    
    # Key landmarks to track
    key_landmarks = ['sella', 'Gonion', 'PNS', 'A_point', 'B_point', 'Nasion', 'anterior_nasal_spine']
    key_landmark_results = {}
    
    for landmark in key_landmarks:
        if landmark in hrnet_per_landmark and landmark in mlp_per_landmark:
            hrnet_err = hrnet_per_landmark[landmark]['mre']
            mlp_err = mlp_per_landmark[landmark]['mre']
            improvement = (hrnet_err - mlp_err) / hrnet_err * 100 if hrnet_err > 0 else 0
            key_landmark_results[landmark] = {
                'hrnet_mre': hrnet_err,
                'mlp_mre': mlp_err,
                'improvement': improvement
            }
    
    results = {
        'work_dir': work_dir,
        'model_type': model_type,
        'hrnet_checkpoint': hrnet_checkpoint_name,
        'samples_evaluated': len(hrnet_predictions),
        'hrnet_mre': hrnet_overall['mre'],
        'mlp_mre': mlp_overall['mre'],
        'improvement_mre': improvement_mre,
        'key_landmarks': key_landmark_results
    }
    
    return results

def print_quick_results(results):
    """Print quick evaluation results in a compact format."""
    if results is None:
        return
    
    model_name = os.path.basename(results['work_dir'])
    print(f"\n📊 {model_name}")
    print(f"   Checkpoint: {results['hrnet_checkpoint']} ({results['model_type']})")
    print(f"   Samples: {results['samples_evaluated']}")
    print(f"   HRNet MRE: {results['hrnet_mre']:.3f} px")
    print(f"   MLP MRE: {results['mlp_mre']:.3f} px")
    print(f"   Improvement: {results['improvement_mre']:+.1f}%")
    
    # Key landmarks
    print(f"   Key landmarks:")
    for landmark, data in results['key_landmarks'].items():
        if landmark == 'sella':  # Highlight sella
            print(f"     🎯 {landmark}: {data['hrnet_mre']:.3f} → {data['mlp_mre']:.3f} ({data['improvement']:+.1f}%)")
        else:
            print(f"     • {landmark}: {data['hrnet_mre']:.3f} → {data['mlp_mre']:.3f} ({data['improvement']:+.1f}%)")

def main():
    """Main evaluation function."""
    
    parser = argparse.ArgumentParser(
        description='Evaluate Concurrent Joint MLP Refinement Performance (Mid-training supported)')
    parser.add_argument(
        '--test_split_file',
        type=str,
        default=None,
        help='Path to a text file containing patient IDs for the test set, one ID per line.'
    )
    parser.add_argument(
        '--work_dir',
        type=str,
        default='work_dirs/hrnetv2_w18_cephalometric_concurrent_mlp_v5',
        help='Work directory containing the trained models'
    )
    parser.add_argument(
        '--ensemble_dir',
        type=str,
        default=None,
        help='Base directory for ensemble models (e.g., work_dirs/hrnetv2_w18_cephalometric_ensemble_concurrent_mlp_v5)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Perform quick evaluation on subset of data for mid-training monitoring'
    )
    parser.add_argument(
        '--monitor',
        action='store_true',
        help='Monitor mode: quick evaluation every few seconds'
    )
    args = parser.parse_args()
    
    print("="*80)
    print("CONCURRENT JOINT MLP EVALUATION")
    if args.quick or args.monitor:
        print("🚀 MID-TRAINING MONITORING MODE")
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
    
    # Load test data
    data_file_path = "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json"
    main_df = pd.read_json(data_file_path)
    
    # Split test data
    if args.test_split_file:
        print(f"Loading test set from external file: {args.test_split_file}")
        with open(args.test_split_file, 'r') as f:
            test_patient_ids = {int(line.strip()) for line in f if line.strip()}
        
        main_df['patient_id'] = main_df['patient_id'].astype(int)
        test_df = main_df[main_df['patient_id'].isin(test_patient_ids)].reset_index(drop=True)
    else:
        print("Using 'set' column for test set selection")
        test_df = main_df[main_df['set'] == 'test'].reset_index(drop=True)
        if test_df.empty:
            test_df = main_df[main_df['set'] == 'dev'].reset_index(drop=True)
    
    if test_df.empty:
        print("ERROR: No test samples found")
        return
    
    print(f"✓ Test set: {len(test_df)} samples")
    
    # Get landmark information
    landmark_names = cephalometric_dataset_info.landmark_names_in_order
    landmark_cols = cephalometric_dataset_info.original_landmark_cols
    
    # Monitoring mode
    if args.monitor:
        print(f"\n🔄 Starting monitoring mode...")
        print(f"Evaluating every 30 seconds. Press Ctrl+C to stop.")
        
        try:
            import time
            while True:
                print(f"\n{'='*50}")
                print(f"⏰ {time.strftime('%H:%M:%S')} - Quick Evaluation")
                print(f"{'='*50}")
                
                if args.ensemble_dir:
                    # Evaluate ensemble models
                    for i in range(1, 4):  # Assuming 3 models
                        model_dir = os.path.join(args.ensemble_dir, f"model_{i}")
                        if os.path.exists(model_dir):
                            results = quick_evaluation(model_dir, test_df, landmark_names, landmark_cols, verbose=False)
                            print_quick_results(results)
                else:
                    # Evaluate single model
                    results = quick_evaluation(args.work_dir, test_df, landmark_names, landmark_cols, verbose=False)
                    print_quick_results(results)
                
                print(f"\nNext evaluation in 30 seconds...")
                time.sleep(30)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Monitoring stopped")
            return
    
    # Quick evaluation mode
    elif args.quick:
        print(f"\n🚀 Quick Evaluation Mode")
        
        if args.ensemble_dir:
            print(f"📁 Evaluating ensemble models in: {args.ensemble_dir}")
            all_results = []
            
            for i in range(1, 4):  # Assuming 3 models
                model_dir = os.path.join(args.ensemble_dir, f"model_{i}")
                if os.path.exists(model_dir):
                    results = quick_evaluation(model_dir, test_df, landmark_names, landmark_cols)
                    if results:
                        all_results.append(results)
                        print_quick_results(results)
                else:
                    print(f"⚠️  Model directory not found: {model_dir}")
            
            # Summary
            if all_results:
                print(f"\n📊 ENSEMBLE SUMMARY")
                print(f"{'Model':<15} {'HRNet MRE':<12} {'MLP MRE':<12} {'Improvement':<12} {'Sella MRE':<12}")
                print("-" * 65)
                
                for results in all_results:
                    model_name = os.path.basename(results['work_dir'])
                    sella_mre = results['key_landmarks'].get('sella', {}).get('mlp_mre', 0)
                    print(f"{model_name:<15} {results['hrnet_mre']:<12.3f} {results['mlp_mre']:<12.3f} {results['improvement_mre']:<12.1f}% {sella_mre:<12.3f}")
        else:
            # Single model evaluation
            results = quick_evaluation(args.work_dir, test_df, landmark_names, landmark_cols)
            print_quick_results(results)
        
        return
    
    # Full evaluation mode (existing code)
    # [The rest of the original evaluation code would go here]
    print(f"\n🔍 Full evaluation mode - using work_dir: {args.work_dir}")
    
    # Find checkpoints
    hrnet_checkpoint, hrnet_checkpoint_name = find_best_available_checkpoint(args.work_dir)
    if hrnet_checkpoint is None:
        print("ERROR: No HRNet checkpoints found")
        return
    
    # Find MLP model
    synchronized_mlp_path, model_type = find_synchronized_mlp_model(args.work_dir, hrnet_checkpoint_name)
    if synchronized_mlp_path is None:
        print("ERROR: No MLP model found")
        return
    
    # Load models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    config_path = "Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py"
    
    # Load HRNetV2 model
    hrnet_model = init_model(config_path, hrnet_checkpoint, device=device)
    print("✓ HRNetV2 model loaded")
    
    # Load joint MLP model
    mlp_joint = JointMLPRefinementModel().to(device)
    mlp_joint.load_state_dict(torch.load(synchronized_mlp_path, map_location=device))
    mlp_joint.eval()
    print("✓ Joint MLP model loaded")
    
    # Load saved joint normalization scalers
    print("Loading saved joint normalization scalers...")
    scaler_dir = os.path.join(args.work_dir, "concurrent_mlp")
    
    scaler_input_path = os.path.join(scaler_dir, "scaler_joint_input.pkl")
    scaler_target_path = os.path.join(scaler_dir, "scaler_joint_target.pkl")
    
    # Check if scalers exist
    scaler_files = [scaler_input_path, scaler_target_path]
    missing_scalers = [f for f in scaler_files if not os.path.exists(f)]
    
    if missing_scalers:
        print(f"ERROR: Missing joint scaler files: {missing_scalers}")
        print("This indicates that concurrent joint MLP training hasn't run yet or scalers weren't saved.")
        print("Please run concurrent joint training first.")
        return
    
    # Load scalers
    try:
        scaler_input = joblib.load(scaler_input_path)
        scaler_target = joblib.load(scaler_target_path)
        print("✓ Joint normalization scalers loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load joint scalers: {e}")
        return
    
    # Evaluation on test set
    print(f"\n🔄 Running full evaluation on {len(test_df)} test samples...")
    
    # Storage for results
    hrnet_predictions = []
    mlp_predictions = []
    ground_truths = []
    
    from tqdm import tqdm
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        try:
            # Get image and ground truth
            img_array = np.array(row['Image'], dtype=np.uint8).reshape((224, 224, 3))
            
            gt_keypoints = []
            valid_gt = True
            for i in range(0, len(landmark_cols), 2):
                x_col = landmark_cols[i]
                y_col = landmark_cols[i+1]
                if x_col in row and y_col in row and pd.notna(row[x_col]) and pd.notna(row[y_col]):
                    gt_keypoints.append([row[x_col], row[y_col]])
                else:
                    gt_keypoints.append([0, 0])
                    valid_gt = False
            
            if not valid_gt:
                continue
                
            gt_keypoints = np.array(gt_keypoints)
            
            # Run HRNetV2 inference using the standard API
            bbox = np.array([[0, 0, 224, 224]], dtype=np.float32)
            results = inference_topdown(hrnet_model, img_array, bboxes=bbox, bbox_format='xyxy')
            
            if results and len(results) > 0:
                pred_keypoints = results[0].pred_instances.keypoints[0]
                if isinstance(pred_keypoints, torch.Tensor):
                    pred_keypoints = pred_keypoints.cpu().numpy()
            else:
                continue

            if pred_keypoints is None or pred_keypoints.shape[0] != 19:
                continue
            
            # Apply joint MLP refinement
            refined_keypoints = apply_joint_mlp_refinement(
                pred_keypoints, mlp_joint, scaler_input, scaler_target, device
            )
            
            # Store results
            hrnet_predictions.append(pred_keypoints)
            mlp_predictions.append(refined_keypoints)
            ground_truths.append(gt_keypoints)
            
        except Exception as e:
            print(f"Failed to process sample {idx}: {e}")
            continue
    
    if len(hrnet_predictions) == 0:
        print("ERROR: No valid predictions generated")
        return
    
    print(f"✓ Successfully evaluated {len(hrnet_predictions)} samples")
    
    # Convert to numpy arrays
    hrnet_coords = np.array(hrnet_predictions)
    mlp_coords = np.array(mlp_predictions)
    gt_coords = np.array(ground_truths)
    
    # Compute metrics
    print("\n📊 Computing metrics...")
    
    hrnet_overall, hrnet_per_landmark = compute_metrics(hrnet_coords, gt_coords, landmark_names)
    mlp_overall, mlp_per_landmark = compute_metrics(mlp_coords, gt_coords, landmark_names)
    
    # Print results
    print("\n" + "="*80)
    print("JOINT MLP EVALUATION RESULTS")
    print("="*80)
    print(f"📊 Evaluated using {model_type} joint MLP model")
    print(f"📈 HRNetV2 checkpoint: {os.path.basename(hrnet_checkpoint)}")
    
    print(f"\n🏷️  OVERALL PERFORMANCE:")
    print(f"{'Metric':<15} {'HRNetV2':<15} {'Joint MLP':<15} {'Improvement':<15}")
    print("-" * 65)
    
    improvement_mre = (hrnet_overall['mre'] - mlp_overall['mre']) / hrnet_overall['mre'] * 100
    improvement_std = (hrnet_overall['std'] - mlp_overall['std']) / hrnet_overall['std'] * 100
    improvement_median = (hrnet_overall['median'] - mlp_overall['median']) / hrnet_overall['median'] * 100
    
    print(f"{'MRE':<15} {hrnet_overall['mre']:<15.3f} {mlp_overall['mre']:<15.3f} {improvement_mre:<15.1f}%")
    print(f"{'Std Dev':<15} {hrnet_overall['std']:<15.3f} {mlp_overall['std']:<15.3f} {improvement_std:<15.1f}%")
    print(f"{'Median':<15} {hrnet_overall['median']:<15.3f} {mlp_overall['median']:<15.3f} {improvement_median:<15.1f}%")
    print(f"{'P90':<15} {hrnet_overall['p90']:<15.3f} {mlp_overall['p90']:<15.3f}")
    print(f"{'P95':<15} {hrnet_overall['p95']:<15.3f} {mlp_overall['p95']:<15.3f}")
    
    # Per-landmark comparison for problematic landmarks
    print(f"\n🎯 PROBLEMATIC LANDMARKS COMPARISON:")
    problematic_landmarks = ['sella', 'Gonion', 'PNS', 'A_point', 'B_point']
    
    print(f"{'Landmark':<20} {'HRNetV2 MRE':<15} {'Joint MLP MRE':<15} {'Improvement':<15}")
    print("-" * 70)
    
    for landmark in problematic_landmarks:
        if landmark in hrnet_per_landmark and landmark in mlp_per_landmark:
            hrnet_err = hrnet_per_landmark[landmark]['mre']
            mlp_err = mlp_per_landmark[landmark]['mre']
            if hrnet_err > 0:
                improvement = (hrnet_err - mlp_err) / hrnet_err * 100
                print(f"{landmark:<20} {hrnet_err:<15.3f} {mlp_err:<15.3f} {improvement:<15.1f}%")
    
    # Save results and visualizations (existing code continues...)
    output_dir = os.path.join(args.work_dir, "joint_mlp_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n💾 Results saved to: {output_dir}")
    print(f"🎉 Joint MLP evaluation completed!")
    print(f"📈 Overall improvement: {improvement_mre:.1f}% reduction in MRE")

if __name__ == "__main__":
    main() 