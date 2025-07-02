#!/usr/bin/env python3
"""
Concurrent Joint MLP Performance Evaluation Script
This script evaluates the performance improvement from joint MLP refinement
trained concurrently with HRNetV2.

The script uses the same random split as the training script (random_state=42):
- 200 samples for test set (evaluation)
- 100 samples for validation set
- Remaining samples for training set

Use --hrnet-only flag to evaluate only HRNet without MLP refinement.
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

def main():
    """Main evaluation function."""
    
    parser = argparse.ArgumentParser(
        description='Evaluate Concurrent Joint MLP Refinement Performance')
    parser.add_argument(
        '--work_dir',
        type=str,
        default='work_dirs/hrnetv2_w18_cephalometric_concurrent_mlp_v5',
        help='Work directory containing the trained models'
    )
    parser.add_argument(
        '--checkpoint_type',
        type=str,
        choices=['best', 'latest', 'epoch'],
        default='best',
        help='Type of checkpoint to evaluate: best (best validation), latest (most recent), or epoch (specific epoch)'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=None,
        help='Specific epoch number to evaluate (only used when checkpoint_type=epoch)'
    )
    parser.add_argument(
        '--hrnet-only',
        action='store_true',
        help='Evaluate only HRNet without MLP refinement'
    )
    parser.add_argument(
        '--data-file',
        type=str,
        default="/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json",
        help='Path to the training data JSON file'
    )
    args = parser.parse_args()
    
    print("="*80)
    if args.hrnet_only:
        print("HRNET ONLY EVALUATION")
    else:
        print("CONCURRENT JOINT MLP REFINEMENT EVALUATION")
    print("="*80)
    print(f"📋 Checkpoint type: {args.checkpoint_type}")
    if args.checkpoint_type == 'epoch':
        print(f"📅 Target epoch: {args.epoch}")
    if args.hrnet_only:
        print("⚠️  MLP evaluation disabled - evaluating HRNet only")
    print()
    
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
    config_path = "Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py"
    
    # Load config
    cfg = Config.fromfile(config_path)
    
    # Find the HRNetV2 checkpoint based on user preference
    hrnet_checkpoint = None
    hrnet_checkpoint_name = None
    
    if args.checkpoint_type == 'best':
        # Look for best validation checkpoint
        hrnet_checkpoint_pattern = os.path.join(args.work_dir, "best_NME_epoch_*.pth")
        hrnet_checkpoints = glob.glob(hrnet_checkpoint_pattern)
        
        if hrnet_checkpoints:
            hrnet_checkpoint = max(hrnet_checkpoints, key=os.path.getctime)
            print(f"✓ Using best validation checkpoint")
        else:
            print("⚠️  No best validation checkpoint found, falling back to latest")
            args.checkpoint_type = 'latest'  # Fallback
    
    if args.checkpoint_type == 'latest':
        # Look for latest checkpoint
        latest_checkpoint = os.path.join(args.work_dir, "latest.pth")
        if os.path.exists(latest_checkpoint):
            hrnet_checkpoint = latest_checkpoint
            print(f"✓ Using latest checkpoint")
        else:
            # Fallback to most recent epoch checkpoint
            hrnet_checkpoint_pattern = os.path.join(args.work_dir, "epoch_*.pth")
            hrnet_checkpoints = glob.glob(hrnet_checkpoint_pattern)
            if hrnet_checkpoints:
                hrnet_checkpoint = max(hrnet_checkpoints, key=lambda x: int(x.split('epoch_')[1].split('.')[0]))
                print(f"✓ Using most recent epoch checkpoint (latest.pth not found)")
    
    if args.checkpoint_type == 'epoch':
        if args.epoch is None:
            print("ERROR: --epoch must be specified when using --checkpoint_type=epoch")
            return
        
        # Look for specific epoch checkpoint
        epoch_checkpoint = os.path.join(args.work_dir, f"epoch_{args.epoch}.pth")
        if os.path.exists(epoch_checkpoint):
            hrnet_checkpoint = epoch_checkpoint
            print(f"✓ Using epoch {args.epoch} checkpoint")
        else:
            print(f"ERROR: Epoch {args.epoch} checkpoint not found: {epoch_checkpoint}")
            return
    
    # Final fallback if no checkpoint found
    if hrnet_checkpoint is None:
        print("ERROR: No suitable HRNetV2 checkpoint found")
        print("Available checkpoint types:")
        
        # Show available checkpoints
        best_checkpoints = glob.glob(os.path.join(args.work_dir, "best_NME_epoch_*.pth"))
        epoch_checkpoints = glob.glob(os.path.join(args.work_dir, "epoch_*.pth"))
        latest_checkpoint = os.path.join(args.work_dir, "latest.pth")
        
        if best_checkpoints:
            print(f"  - Best validation: {len(best_checkpoints)} checkpoint(s)")
        if os.path.exists(latest_checkpoint):
            print(f"  - Latest: available")
        if epoch_checkpoints:
            epochs = sorted([int(f.split('epoch_')[1].split('.')[0]) for f in epoch_checkpoints])
            print(f"  - Epoch-specific: epochs {epochs[0]}-{epochs[-1]} ({len(epochs)} total)")
        
        return
    
    hrnet_checkpoint_name = os.path.basename(hrnet_checkpoint)
    print(f"✓ Using HRNetV2 checkpoint: {hrnet_checkpoint_name}")
    
    # Load checkpoint mapping to find synchronized MLP model (skip if hrnet-only)
    synchronized_mlp_path = None
    model_type = "hrnet_only" if args.hrnet_only else "unknown"
    
    if not args.hrnet_only:
        mlp_dir = os.path.join(args.work_dir, "concurrent_mlp")
        mapping_file = os.path.join(mlp_dir, "checkpoint_mlp_mapping.json")
    
        # Try to find synchronized model first
        if os.path.exists(mapping_file):
            try:
                import json
                with open(mapping_file, 'r') as f:
                    checkpoint_mapping = json.load(f)
                
                # Look for synchronized MLP model for this HRNet checkpoint
                if hrnet_checkpoint_name in checkpoint_mapping:
                    synchronized_mlp_path = checkpoint_mapping[hrnet_checkpoint_name]
                    if os.path.exists(synchronized_mlp_path):
                        print(f"✓ Found synchronized MLP model: {os.path.basename(synchronized_mlp_path)}")
                        model_type = f"synchronized_with_{hrnet_checkpoint_name}"
                    else:
                        print(f"⚠️  Mapped MLP model not found: {synchronized_mlp_path}")
                        synchronized_mlp_path = None
                else:
                    print(f"⚠️  No synchronized MLP model found for checkpoint: {hrnet_checkpoint_name}")
                    print(f"Available mappings: {list(checkpoint_mapping.keys())}")
                    
            except Exception as e:
                print(f"⚠️  Failed to load checkpoint mapping: {e}")
        else:
            print(f"💡 Checkpoint mapping file not found - likely early training stage")
            print(f"   Will use epoch-based MLP model matching strategy")
        
        # Fallback to existing logic if no synchronized model found
        if synchronized_mlp_path is None:
            print("🔄 Using epoch-based MLP model matching...")
            
            # Try to extract epoch number from HRNet checkpoint for better matching
            hrnet_epoch = None
            if "epoch_" in hrnet_checkpoint_name:
                try:
                    hrnet_epoch = int(hrnet_checkpoint_name.split("epoch_")[1].split(".")[0])
                    print(f"💡 HRNet checkpoint is from epoch {hrnet_epoch}")
                except:
                    pass
            
            # First, try to find MLP model from the same epoch
            if hrnet_epoch is not None:
                epoch_mlp_path = os.path.join(mlp_dir, f"mlp_joint_epoch_{hrnet_epoch}.pth")
                if os.path.exists(epoch_mlp_path):
                    synchronized_mlp_path = epoch_mlp_path
                    print(f"✓ Found matching epoch MLP model: {os.path.basename(epoch_mlp_path)}")
                    model_type = f"epoch_matched_{hrnet_epoch}"
                else:
                    print(f"⚠️  No MLP model found for epoch {hrnet_epoch}")
            
            # If no epoch match, fall back to other strategies
            if synchronized_mlp_path is None:
                # Check for final model
                mlp_joint_path = os.path.join(mlp_dir, "mlp_joint_final.pth")
                
                if os.path.exists(mlp_joint_path):
                    synchronized_mlp_path = mlp_joint_path
                    print(f"✓ Found final joint MLP model: {mlp_joint_path}")
                    model_type = "final_fallback"
                else:
                    # Try latest model
                    mlp_joint_latest = os.path.join(mlp_dir, "mlp_joint_latest.pth")
                    
                    if os.path.exists(mlp_joint_latest):
                        synchronized_mlp_path = mlp_joint_latest
                        print(f"✓ Found latest joint MLP model: {mlp_joint_latest}")
                        model_type = "latest_fallback"
                    else:
                        # Try to find any epoch-specific models
                        epoch_models = glob.glob(os.path.join(mlp_dir, "mlp_joint_epoch_*.pth"))
                        if epoch_models:
                            # Get the latest epoch model
                            latest_joint_model = max(epoch_models, key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
                            epoch_num = latest_joint_model.split('_epoch_')[1].split('.')[0]
                            
                            synchronized_mlp_path = latest_joint_model
                            print(f"✓ Found epoch {epoch_num} joint MLP model: {latest_joint_model}")
                            model_type = f"epoch_{epoch_num}_fallback"
                            
                            # Warn if there's a significant epoch mismatch
                            if hrnet_epoch is not None and abs(int(epoch_num) - hrnet_epoch) > 2:
                                print(f"⚠️  Warning: Using MLP from epoch {epoch_num} with HRNet from epoch {hrnet_epoch}")
                                print(f"   Consider waiting for more training or using synchronized checkpoints")
                        else:
                            print("ERROR: No joint MLP model found.")
                            print(f"Searched in: {mlp_dir}")
                            print("Available files:")
                            if os.path.exists(mlp_dir):
                                for file in os.listdir(mlp_dir):
                                    print(f"  - {file}")
                            else:
                                print("  MLP directory does not exist")
                            print("\nTip: Make sure concurrent joint training is running and has completed at least one epoch.")
                            return
    
    # Load models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    # Load HRNetV2 model
    hrnet_model = init_model(config_path, hrnet_checkpoint, device=device)
    print("✓ HRNetV2 model loaded")
    
    # Load joint MLP model (skip if hrnet-only)
    mlp_joint = None
    scaler_input = None
    scaler_target = None
    
    if not args.hrnet_only:
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
    
    # Load test data
    data_file_path = args.data_file
    main_df = pd.read_json(data_file_path)
    
    # Use same random split logic as training script (random seed 42)
    print("Using same random split as training: 200 test samples, 100 validation samples, rest for training")
    
    # Check if we have enough samples
    total_samples = len(main_df)
    required_samples = 300  # 200 test + 100 validation
    
    if total_samples < required_samples:
        print(f"ERROR: Not enough samples for random split.")
        print(f"Total samples: {total_samples}, Required: {required_samples} (200 test + 100 validation)")
        return
        
    print(f"Total samples available: {total_samples}")
    
    # Shuffle the entire dataframe with same random state as training (42)
    shuffled_df = main_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split randomly: first 200 for test, next 100 for validation, rest for training
    test_df = shuffled_df.iloc[:200].reset_index(drop=True)
    val_df = shuffled_df.iloc[200:300].reset_index(drop=True)
    train_df = shuffled_df.iloc[300:].reset_index(drop=True)
    
    print(f"Random split completed (same as training):")
    print(f"  • Test set: {len(test_df)} samples")
    print(f"  • Validation set: {len(val_df)} samples") 
    print(f"  • Training set: {len(train_df)} samples")
    
    if test_df.empty:
        print("ERROR: No test samples found")
        return
    
    print(f"✓ Evaluating on {len(test_df)} test samples (matches training split)")
    
    # Get landmark information
    landmark_names = cephalometric_dataset_info.landmark_names_in_order
    landmark_cols = cephalometric_dataset_info.original_landmark_cols
    
    # Evaluation on test set
    print(f"\n🔄 Running evaluation on {len(test_df)} test samples...")
    
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
            
            # Apply joint MLP refinement (skip if hrnet-only)
            if not args.hrnet_only:
                refined_keypoints = apply_joint_mlp_refinement(
                    pred_keypoints, mlp_joint, scaler_input, scaler_target, device
                )
                mlp_predictions.append(refined_keypoints)
            
            # Store results
            hrnet_predictions.append(pred_keypoints)
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
    gt_coords = np.array(ground_truths)
    
    # Compute metrics
    print("\n📊 Computing metrics...")
    
    hrnet_overall, hrnet_per_landmark = compute_metrics(hrnet_coords, gt_coords, landmark_names)
    
    if not args.hrnet_only:
        mlp_coords = np.array(mlp_predictions)
        mlp_overall, mlp_per_landmark = compute_metrics(mlp_coords, gt_coords, landmark_names)
    else:
        mlp_coords = None
        mlp_overall = None
        mlp_per_landmark = None
    
    # Print results
    print("\n" + "="*80)
    if args.hrnet_only:
        print("HRNET ONLY EVALUATION RESULTS")
    else:
        print("JOINT MLP EVALUATION RESULTS")
    print("="*80)
    
    if not args.hrnet_only:
        print(f"📊 Evaluated using {model_type} joint MLP model")
    print(f"📈 HRNetV2 checkpoint: {hrnet_checkpoint_name} ({args.checkpoint_type})")
    print(f"🎯 Total samples evaluated: {len(hrnet_predictions)}")
    
    if args.checkpoint_type == 'epoch' and args.epoch:
        print(f"📅 Specific epoch evaluated: {args.epoch}")
    elif args.checkpoint_type == 'latest':
        print(f"🔄 Latest training checkpoint used")
    elif args.checkpoint_type == 'best':
        print(f"🏆 Best validation checkpoint used")
    
    if args.hrnet_only:
        print(f"\n🏷️  HRNET PERFORMANCE:")
        print(f"{'Metric':<15} {'HRNetV2':<15}")
        print("-" * 30)
        
        print(f"{'MRE':<15} {hrnet_overall['mre']:<15.3f}")
        print(f"{'Std Dev':<15} {hrnet_overall['std']:<15.3f}")
        print(f"{'Median':<15} {hrnet_overall['median']:<15.3f}")
        print(f"{'P90':<15} {hrnet_overall['p90']:<15.3f}")
        print(f"{'P95':<15} {hrnet_overall['p95']:<15.3f}")
        
        # Per-landmark results for problematic landmarks
        print(f"\n🎯 PROBLEMATIC LANDMARKS:")
        problematic_landmarks = ['sella', 'Gonion', 'PNS', 'A_point', 'B_point']
        
        print(f"{'Landmark':<20} {'HRNetV2 MRE':<15}")
        print("-" * 35)
        
        for landmark in problematic_landmarks:
            if landmark in hrnet_per_landmark:
                hrnet_err = hrnet_per_landmark[landmark]['mre']
                print(f"{landmark:<20} {hrnet_err:<15.3f}")
    else:
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
    
    # Save results
    if args.hrnet_only:
        output_dir = os.path.join(args.work_dir, "hrnet_evaluation")
    else:
        output_dir = os.path.join(args.work_dir, "joint_mlp_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    if args.hrnet_only:
        results_summary = {
            'hrnet_overall': hrnet_overall,
            'total_samples': len(hrnet_predictions),
            'model_type': model_type
        }
    else:
        results_summary = {
            'hrnet_overall': hrnet_overall,
            'mlp_overall': mlp_overall,
            'improvement_mre': improvement_mre,
            'improvement_std': improvement_std,
            'improvement_median': improvement_median,
            'total_samples': len(hrnet_predictions),
            'model_type': model_type
        }
    
    # Save per-landmark comparison
    per_landmark_comparison = []
    for landmark in landmark_names:
        if landmark in hrnet_per_landmark:
            hrnet_err = hrnet_per_landmark[landmark]['mre']
            
            if args.hrnet_only:
                per_landmark_comparison.append({
                    'landmark': landmark,
                    'hrnet_mre': hrnet_err,
                    'hrnet_std': hrnet_per_landmark[landmark]['std']
                })
            else:
                if landmark in mlp_per_landmark:
                    mlp_err = mlp_per_landmark[landmark]['mre']
                    improvement = (hrnet_err - mlp_err) / hrnet_err * 100 if hrnet_err > 0 else 0
                    
                    per_landmark_comparison.append({
                        'landmark': landmark,
                        'hrnet_mre': hrnet_err,
                        'mlp_mre': mlp_err,
                        'improvement_percent': improvement,
                        'hrnet_std': hrnet_per_landmark[landmark]['std'],
                        'mlp_std': mlp_per_landmark[landmark]['std']
                    })
    
    comparison_df = pd.DataFrame(per_landmark_comparison)
    comparison_df.to_csv(os.path.join(output_dir, "per_landmark_comparison.csv"), index=False)
    
    # Create visualization
    if args.hrnet_only:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Overall performance
        metrics = ['MRE', 'Std Dev', 'Median', 'P90', 'P95']
        values = [hrnet_overall['mre'], hrnet_overall['std'], hrnet_overall['median'], 
                 hrnet_overall['p90'], hrnet_overall['p95']]
        
        ax1.bar(metrics, values, alpha=0.7, color='skyblue')
        ax1.set_ylabel('Error (pixels)')
        ax1.set_title('HRNetV2 Performance Metrics')
        ax1.grid(True, alpha=0.3)
        
        # Error distribution
        hrnet_errors = np.sqrt(np.sum((hrnet_coords - gt_coords)**2, axis=2)).flatten()
        valid_mask = (gt_coords.reshape(-1, 2)[:, 0] > 0) & (gt_coords.reshape(-1, 2)[:, 1] > 0)
        hrnet_errors = hrnet_errors[valid_mask]
        
        ax2.hist(hrnet_errors, bins=50, alpha=0.7, color='skyblue')
        ax2.set_xlabel('Radial Error (pixels)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('HRNetV2 Error Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "hrnet_evaluation_results.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall comparison
        methods = ['HRNetV2', 'Joint MLP']
        mres = [hrnet_overall['mre'], mlp_overall['mre']]
        stds = [hrnet_overall['std'], mlp_overall['std']]
        
        ax1.bar(methods, mres, yerr=stds, capsize=5, alpha=0.7, color=['skyblue', 'lightcoral'])
        ax1.set_ylabel('Mean Radial Error (pixels)')
        ax1.set_title('Overall Performance Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Add improvement percentage
        ax1.text(1, mlp_overall['mre'] + mlp_overall['std'] + 0.1, 
                 f'{improvement_mre:.1f}% improvement', ha='center', va='bottom', 
                 fontsize=10, color='green' if improvement_mre > 0 else 'red', fontweight='bold')
        
        # Per-landmark improvements
        landmarks_subset = comparison_df.head(10)  # Top 10 landmarks
        ax2.barh(landmarks_subset['landmark'], landmarks_subset['improvement_percent'], 
                 color=['green' if x > 0 else 'red' for x in landmarks_subset['improvement_percent']])
        ax2.set_xlabel('Improvement (%)')
        ax2.set_title('Per-Landmark Improvement')
        ax2.grid(True, alpha=0.3)
        
        # Error distribution comparison
        hrnet_errors = np.sqrt(np.sum((hrnet_coords - gt_coords)**2, axis=2)).flatten()
        mlp_errors = np.sqrt(np.sum((mlp_coords - gt_coords)**2, axis=2)).flatten()
        
        # Remove invalid landmarks (where gt is [0,0])
        valid_mask = (gt_coords.reshape(-1, 2)[:, 0] > 0) & (gt_coords.reshape(-1, 2)[:, 1] > 0)
        hrnet_errors = hrnet_errors[valid_mask]
        mlp_errors = mlp_errors[valid_mask]
        
        ax3.hist([hrnet_errors, mlp_errors], bins=50, alpha=0.7, 
                 label=['HRNetV2', 'Joint MLP'], color=['skyblue', 'lightcoral'])
        ax3.set_xlabel('Radial Error (pixels)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Error Distribution Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Scatter plot: HRNet vs MLP errors
        sample_indices = np.random.choice(len(hrnet_errors), min(1000, len(hrnet_errors)), replace=False)
        ax4.scatter(hrnet_errors[sample_indices], mlp_errors[sample_indices], 
                    alpha=0.5, color='purple')
        
        # Add diagonal line
        max_error = max(np.max(hrnet_errors), np.max(mlp_errors))
        ax4.plot([0, max_error], [0, max_error], 'r--', alpha=0.8, label='No improvement line')
        ax4.set_xlabel('HRNetV2 Error (pixels)')
        ax4.set_ylabel('Joint MLP Error (pixels)')
        ax4.set_title('Error Correlation (Sample)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "joint_mlp_evaluation_results.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\n💾 Results saved to: {output_dir}")
    print(f"   - Per-landmark comparison: per_landmark_comparison.csv")
    if args.hrnet_only:
        print(f"   - Visualization: hrnet_evaluation_results.png")
        print(f"\n🎉 HRNet evaluation completed!")
        print(f"📈 Overall MRE: {hrnet_overall['mre']:.3f} pixels")
        print(f"🎯 Baseline performance established")
    else:
        print(f"   - Visualization: joint_mlp_evaluation_results.png")
        print(f"\n🎉 Joint MLP evaluation completed!")
        print(f"📈 Overall improvement: {improvement_mre:.1f}% reduction in MRE")
        print(f"🎯 Joint model captures cross-correlations between landmarks")
        print(f"🔧 Evaluated using: {model_type} joint MLP model")
        
        if model_type == "latest_fallback":
            print("💡 Note: Training is likely still in progress. Final results may differ.")
        elif "epoch_" in model_type:
            print("💡 Note: Using intermediate checkpoint. Final results may differ.")

if __name__ == "__main__":
    main() 