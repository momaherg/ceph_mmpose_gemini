#!/usr/bin/env python3
"""
Concurrent Joint MLP Performance Evaluation Script
This script evaluates the performance improvement from joint MLP refinement
trained concurrently with HRNetV2.

IMPORTANT: Uses TRUE batch inference (same as training hook) for consistency.
This ensures evaluation results match the inference method used during training.
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
    """Joint MLP model for landmark coordinate refinement with adaptive selection."""
    def __init__(self, input_dim=38, hidden_dim=500, output_dim=38):
        super(JointMLPRefinementModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Main refinement network
        self.refinement_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Selection/gating network - learns when to trust HRNet vs MLP
        # Outputs per-coordinate selection weights (38 weights for 38 coordinates)
        self.selection_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()  # Output between 0 and 1 for each coordinate
        )
        
        # Residual projection (if dimensions don't match)
        self.residual_proj = None
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: HRNet predictions [batch_size, 38]
            
        Returns:
            Adaptively selected coordinates [batch_size, 38]
        """
        # Get MLP refinement predictions
        mlp_refinement = self.refinement_net(x)
        
        # Add residual connection to MLP predictions
        if self.residual_proj is not None:
            residual = self.residual_proj(x)
        else:
            residual = x
        
        mlp_predictions = mlp_refinement + 0.1 * residual
        
        # Get selection weights (0 = use HRNet, 1 = use MLP)
        selection_weights = self.selection_net(x)
        
        # Adaptive combination: weighted average of HRNet and MLP predictions
        # output = (1 - weight) * hrnet + weight * mlp
        adaptive_output = (1 - selection_weights) * x + selection_weights * mlp_predictions
        
        # Store selection weights for analysis (optional)
        self.last_selection_weights = selection_weights
        
        return adaptive_output

def batch_hrnet_inference(images_batch, model, device):
    """Run TRUE batch inference using the same method as training hook."""
    try:
        if len(images_batch) == 0:
            return []
        
        # Convert to tensor batch [batch_size, C, H, W]
        images_array = np.stack(images_batch)  # [N, H, W, C]
        images_tensor = torch.from_numpy(images_array).permute(0, 3, 1, 2).float()  # [N, C, H, W]
        images_tensor = images_tensor.to(device)
        
        # Normalize images (ImageNet normalization)
        mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(device)
        images_tensor = (images_tensor - mean) / std
        
        # Run TRUE batch inference using model directly
        with torch.no_grad():
            # Get features from backbone
            if hasattr(model, 'extract_feat'):
                features = model.extract_feat(images_tensor)
            else:
                features = model.backbone(images_tensor)
            
            # Get predictions from head
            if hasattr(model, 'head'):
                head_outputs = model.head(features)
                
                # Extract heatmaps and convert to coordinates
                if isinstance(head_outputs, (list, tuple)):
                    heatmaps = head_outputs[0]
                else:
                    heatmaps = head_outputs
                
                # Decode heatmaps to coordinates
                batch_size, num_keypoints, heatmap_h, heatmap_w = heatmaps.shape
                
                # Find max locations in heatmaps
                heatmaps_reshaped = heatmaps.view(batch_size, num_keypoints, -1)
                max_vals, max_indices = torch.max(heatmaps_reshaped, dim=2)
                
                # Convert indices to coordinates
                max_indices_y = max_indices // heatmap_w
                max_indices_x = max_indices % heatmap_w
                
                # Scale to original image size
                scale_x = 224.0 / heatmap_w
                scale_y = 224.0 / heatmap_h
                
                pred_coords = torch.stack([
                    max_indices_x.float() * scale_x,
                    max_indices_y.float() * scale_y
                ], dim=-1)  # [batch_size, num_keypoints, 2]
                
                return pred_coords.cpu().numpy()
            else:
                return None
                
    except Exception as e:
        print(f"Batch HRNet inference failed: {e}")
        return None

def apply_joint_mlp_refinement(predictions, mlp_joint, scaler_input, scaler_target, device):
    """Apply joint MLP refinement to predictions and return selection weights."""
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
            # Get selection weights if available
            selection_weights = None
            if hasattr(mlp_joint, 'last_selection_weights'):
                selection_weights = mlp_joint.last_selection_weights.cpu().numpy().flatten()
        
        # Denormalize outputs
        refined_flat = scaler_target.inverse_transform(refined_scaled).flatten()
        
        # Reshape back to [19, 2] format
        refined_coords = refined_flat.reshape(19, 2)
        
        return refined_coords, selection_weights
        
    except Exception as e:
        print(f"Joint MLP refinement failed: {e}")
        return predictions, None

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
    args = parser.parse_args()
    
    print("="*80)
    print("CONCURRENT JOINT MLP REFINEMENT EVALUATION")
    print("✅ Using TRUE batch inference (consistent with training hook)")
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
    
    # Configuration
    config_path = "Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py"
    
    # Load config
    cfg = Config.fromfile(config_path)
    
    # Find the best HRNetV2 checkpoint
    hrnet_checkpoint_pattern = os.path.join(args.work_dir, "best_NME_epoch_*.pth")
    hrnet_checkpoints = glob.glob(hrnet_checkpoint_pattern)
    
    if not hrnet_checkpoints:
        hrnet_checkpoint_pattern = os.path.join(args.work_dir, "epoch_*.pth")
        hrnet_checkpoints = glob.glob(hrnet_checkpoint_pattern)
    
    if not hrnet_checkpoints:
        print("ERROR: No HRNetV2 checkpoints found")
        return
    
    hrnet_checkpoint = max(hrnet_checkpoints, key=os.path.getctime)
    hrnet_checkpoint_name = os.path.basename(hrnet_checkpoint)
    print(f"✓ Using HRNetV2 checkpoint: {hrnet_checkpoint}")
    
    # Load checkpoint mapping to find synchronized MLP model
    mlp_dir = os.path.join(args.work_dir, "concurrent_mlp")
    mapping_file = os.path.join(mlp_dir, "checkpoint_mlp_mapping.json")
    
    synchronized_mlp_path = None
    model_type = "unknown"
    
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
    
    # Load joint MLP model
    mlp_joint = JointMLPRefinementModel().to(device)
    mlp_joint.load_state_dict(torch.load(synchronized_mlp_path, map_location=device))
    mlp_joint.eval()
    print("✓ Joint MLP model loaded")
    
    # Load test data
    data_file_path = "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json"
    main_df = pd.read_json(data_file_path)
    
    # Split test data (same logic as training scripts)
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
    
    print(f"✓ Evaluating on {len(test_df)} test samples")
    
    # Get landmark information
    landmark_names = cephalometric_dataset_info.landmark_names_in_order
    landmark_cols = cephalometric_dataset_info.original_landmark_cols
    
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
    
    # Evaluation on test set using TRUE batch inference (same as training hook)
    print(f"\n🔄 Running evaluation using TRUE batch inference (consistent with training)...")
    print(f"📊 Processing {len(test_df)} test samples in batches...")
    
    # Storage for results
    hrnet_predictions = []
    mlp_predictions = []
    ground_truths = []
    all_selection_weights = []
    
    from tqdm import tqdm
    
    # Batch processing parameters
    EVAL_BATCH_SIZE = 64  # Smaller than training for evaluation
    
    # Prepare all data first
    all_images = []
    all_gt_keypoints = []
    valid_indices = []
    
    print("📋 Preparing evaluation data...")
    for idx, row in test_df.iterrows():
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
            
            if valid_gt:
                all_images.append(img_array)
                all_gt_keypoints.append(np.array(gt_keypoints))
                valid_indices.append(idx)
            
        except Exception as e:
            continue
    
    if not all_images:
        print("ERROR: No valid samples for evaluation")
        return
    
    print(f"✓ Prepared {len(all_images)} valid samples for batch evaluation")
    
    # Process in batches using TRUE batch inference
    for batch_start in tqdm(range(0, len(all_images), EVAL_BATCH_SIZE), desc="Batch Evaluation"):
        batch_end = min(batch_start + EVAL_BATCH_SIZE, len(all_images))
        
        # Get batch data
        batch_images = all_images[batch_start:batch_end]
        batch_gt = all_gt_keypoints[batch_start:batch_end]
        
        try:
            # Run TRUE batch HRNet inference (same method as training hook)
            batch_hrnet_preds = batch_hrnet_inference(batch_images, hrnet_model, device)
            
            if batch_hrnet_preds is None:
                # Fallback to individual inference if batch fails
                print(f"⚠️  Batch inference failed for batch {batch_start//EVAL_BATCH_SIZE + 1}, using fallback...")
                
                for img, gt_kpts in zip(batch_images, batch_gt):
                    try:
                        bbox = np.array([[0, 0, 224, 224]], dtype=np.float32)
                        results = inference_topdown(hrnet_model, img, bboxes=bbox, bbox_format='xyxy')
                        
                        if results and len(results) > 0:
                            pred_kpts = results[0].pred_instances.keypoints[0]
                            if isinstance(pred_kpts, torch.Tensor):
                                pred_kpts = pred_kpts.cpu().numpy()
                                
                            if pred_kpts is not None and pred_kpts.shape[0] == 19:
                                # Apply MLP refinement
                                refined_kpts, selection_weights = apply_joint_mlp_refinement(
                                    pred_kpts, mlp_joint, scaler_input, scaler_target, device
                                )
                                
                                hrnet_predictions.append(pred_kpts)
                                mlp_predictions.append(refined_kpts)
                                ground_truths.append(gt_kpts)
                                if selection_weights is not None:
                                    all_selection_weights.append(selection_weights)
                    except:
                        continue
                continue
            
            # Process batch predictions
            for i, (pred_kpts, gt_kpts) in enumerate(zip(batch_hrnet_preds, batch_gt)):
                if pred_kpts.shape[0] == 19:  # Ensure correct number of landmarks
                    try:
                        # Apply joint MLP refinement
                        refined_kpts, selection_weights = apply_joint_mlp_refinement(
                            pred_kpts, mlp_joint, scaler_input, scaler_target, device
                        )
                        
                        # Store results
                        hrnet_predictions.append(pred_kpts)
                        mlp_predictions.append(refined_kpts)
                        ground_truths.append(gt_kpts)
                        if selection_weights is not None:
                            all_selection_weights.append(selection_weights)
                            
                    except Exception as e:
                        continue
        
        except Exception as e:
            print(f"Batch {batch_start//EVAL_BATCH_SIZE + 1} failed: {e}")
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
    
    # Analyze selection weights if available
    if all_selection_weights:
        print(f"\n🔍 ADAPTIVE SELECTION ANALYSIS:")
        print(f"{'='*70}")
        
        # Convert to numpy array and reshape
        selection_weights_array = np.array(all_selection_weights)  # [N, 38]
        avg_selection_weights = np.mean(selection_weights_array, axis=0)  # [38]
        
        # Overall statistics
        overall_mlp_usage = np.mean(avg_selection_weights)
        print(f"📊 Overall MLP usage: {overall_mlp_usage:.3f} (0=always HRNet, 1=always MLP)")
        
        # Per-landmark statistics (reshape to [19, 2] for landmarks)
        landmark_weights = avg_selection_weights.reshape(19, 2)  # [19 landmarks, 2 coords (x,y)]
        landmark_avg_weights = np.mean(landmark_weights, axis=1)  # Average over x,y
        
        # Sort landmarks by MLP preference
        sorted_indices = np.argsort(landmark_avg_weights)[::-1]
        
        print(f"\n🎯 Landmarks by MLP preference (highest to lowest):")
        print(f"{'Landmark':<20} {'Avg Weight':<12} {'X Weight':<12} {'Y Weight':<12} {'Preference':<15}")
        print("-" * 75)
        
        for idx in sorted_indices[:10]:  # Top 10
            landmark = landmark_names[idx]
            avg_w = landmark_avg_weights[idx]
            x_w = landmark_weights[idx, 0]
            y_w = landmark_weights[idx, 1]
            
            if avg_w > 0.7:
                preference = "Strong MLP"
            elif avg_w > 0.5:
                preference = "Moderate MLP"
            elif avg_w > 0.3:
                preference = "Mixed"
            else:
                preference = "Strong HRNet"
                
            print(f"{landmark:<20} {avg_w:<12.3f} {x_w:<12.3f} {y_w:<12.3f} {preference:<15}")
        
        # Correlation with error improvement
        print(f"\n📈 Selection weights vs. improvement correlation:")
        improvements = []
        weights = []
        
        for i, landmark in enumerate(landmark_names):
            if landmark in hrnet_per_landmark and landmark in mlp_per_landmark:
                hrnet_err = hrnet_per_landmark[landmark]['mre']
                mlp_err = mlp_per_landmark[landmark]['mre']
                if hrnet_err > 0:
                    improvement = (hrnet_err - mlp_err) / hrnet_err * 100
                    improvements.append(improvement)
                    weights.append(landmark_avg_weights[i])
        
        if improvements:
            correlation = np.corrcoef(weights, improvements)[0, 1]
            print(f"Correlation between MLP usage and improvement: {correlation:.3f}")
            
            if correlation > 0.3:
                print("✅ Positive correlation: Model learns to use MLP more where it helps more")
            elif correlation < -0.3:
                print("⚠️  Negative correlation: Model may be over-conservative with MLP usage")
            else:
                print("🔄 Weak correlation: Selection may be based on other factors")
    
    # Save results
    output_dir = os.path.join(args.work_dir, "joint_mlp_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
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
        if landmark in hrnet_per_landmark and landmark in mlp_per_landmark:
            hrnet_err = hrnet_per_landmark[landmark]['mre']
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