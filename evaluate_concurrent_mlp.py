#!/usr/bin/env python3
"""
Concurrent MLP Performance Evaluation Script
This script evaluates the performance improvement from MLP refinement
trained concurrently with HRNetV2.
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
from mmpose.apis import init_model
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

class MLPRefinementModel(nn.Module):
    """MLP model for landmark coordinate refinement."""
    def __init__(self, input_dim=19, hidden_dim=500, output_dim=19):
        super(MLPRefinementModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)

def run_hrnet_inference(model, img_array, device):
    """Run HRNetV2 inference on a single image."""
    try:
        import cv2
        from mmpose.structures import PoseDataSample
        from mmengine.structures import InstanceData
        
        # Resize image to model input size (384x384)
        input_size = (384, 384)
        img_resized = cv2.resize(img_array, input_size)
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        # Add batch dimension and move to device
        img_batch = img_tensor.unsqueeze(0).to(device)
        
        # Create data sample with required metadata
        data_sample = PoseDataSample()
        instance_data = InstanceData()
        instance_data.bboxes = torch.tensor([[0, 0, input_size[0], input_size[1]]], dtype=torch.float32)
        instance_data.bbox_scores = torch.tensor([1.0], dtype=torch.float32)
        data_sample.gt_instances = instance_data
        
        # Add metadata
        center = np.array([img_array.shape[1]/2, img_array.shape[0]/2])
        scale = np.array([img_array.shape[1], img_array.shape[0]])
        
        data_sample.set_metainfo({
            'flip_indices': list(range(19)),
            'input_size': input_size,
            'center': center,
            'scale': scale,
            'input_center': center,
            'input_scale': scale
        })
        
        # Run inference
        with torch.no_grad():
            results = model(img_batch, [data_sample], mode='predict')
        
        if results and len(results) > 0 and hasattr(results[0], 'pred_instances'):
            pred_keypoints = results[0].pred_instances.keypoints[0]
            
            # Scale back to original image size (224x224)
            scale_x = 224.0 / input_size[0]
            scale_y = 224.0 / input_size[1]
            
            if isinstance(pred_keypoints, torch.Tensor):
                scale_tensor = torch.tensor([scale_x, scale_y]).to(pred_keypoints.device)
                pred_keypoints = pred_keypoints * scale_tensor
                return pred_keypoints.cpu().numpy()
            else:
                pred_keypoints = np.array(pred_keypoints)
                pred_keypoints[:, 0] *= scale_x
                pred_keypoints[:, 1] *= scale_y
                return pred_keypoints
        else:
            return None
            
    except Exception as e:
        print(f"Inference failed: {e}")
        return None

def apply_mlp_refinement(predictions_x, predictions_y, mlp_x, mlp_y, scaler_x_input, scaler_x_target, scaler_y_input, scaler_y_target, device):
    """Apply MLP refinement to predictions."""
    try:
        # Normalize input predictions
        pred_x_scaled = scaler_x_input.transform(predictions_x.reshape(1, -1))
        pred_y_scaled = scaler_y_input.transform(predictions_y.reshape(1, -1))
        
        # Convert to tensors
        pred_x_tensor = torch.FloatTensor(pred_x_scaled).to(device)
        pred_y_tensor = torch.FloatTensor(pred_y_scaled).to(device)
        
        # Apply MLP refinement
        with torch.no_grad():
            refined_x_scaled = mlp_x(pred_x_tensor).cpu().numpy()
            refined_y_scaled = mlp_y(pred_y_tensor).cpu().numpy()
        
        # Denormalize outputs
        refined_x = scaler_x_target.inverse_transform(refined_x_scaled).flatten()
        refined_y = scaler_y_target.inverse_transform(refined_y_scaled).flatten()
        
        return refined_x, refined_y
        
    except Exception as e:
        print(f"MLP refinement failed: {e}")
        return predictions_x, predictions_y

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
        description='Evaluate Concurrent MLP Refinement Performance')
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
    print("CONCURRENT MLP REFINEMENT EVALUATION")
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
    print(f"✓ Using HRNetV2 checkpoint: {hrnet_checkpoint}")
    
    # Check for MLP models
    mlp_dir = os.path.join(args.work_dir, "concurrent_mlp")
    mlp_x_path = os.path.join(mlp_dir, "mlp_x_final.pth")
    mlp_y_path = os.path.join(mlp_dir, "mlp_y_final.pth")
    
    if not (os.path.exists(mlp_x_path) and os.path.exists(mlp_y_path)):
        print("ERROR: MLP models not found. Make sure concurrent training completed successfully.")
        print(f"Looking for: {mlp_x_path} and {mlp_y_path}")
        return
    
    print(f"✓ Found MLP models: {mlp_x_path}, {mlp_y_path}")
    
    # Load models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    # Load HRNetV2 model
    hrnet_model = init_model(config_path, hrnet_checkpoint, device=device)
    print("✓ HRNetV2 model loaded")
    
    # Load MLP models
    mlp_x = MLPRefinementModel().to(device)
    mlp_y = MLPRefinementModel().to(device)
    
    mlp_x.load_state_dict(torch.load(mlp_x_path, map_location=device))
    mlp_y.load_state_dict(torch.load(mlp_y_path, map_location=device))
    
    mlp_x.eval()
    mlp_y.eval()
    print("✓ MLP models loaded")
    
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
    
    # We need to compute normalization statistics from training data for MLP
    print("Computing normalization statistics from training data...")
    
    # Load training data for normalization
    if args.test_split_file:
        remaining_df = main_df[~main_df['patient_id'].isin(test_patient_ids)]
        if len(remaining_df) >= 100:
            val_df = remaining_df.sample(n=100, random_state=42)
            train_df = remaining_df.drop(val_df.index).reset_index(drop=True)
        else:
            train_df = remaining_df
    else:
        train_df = main_df[main_df['set'] == 'train'].reset_index(drop=True)
    
    # Generate normalization data by running inference on a subset of training data
    print("Generating normalization statistics (this may take a moment)...")
    norm_sample_size = min(200, len(train_df))  # Use subset for efficiency
    norm_preds_x = []
    norm_preds_y = []
    norm_gts_x = []
    norm_gts_y = []
    
    for idx, row in train_df.head(norm_sample_size).iterrows():
        try:
            img_array = np.array(row['Image'], dtype=np.uint8).reshape((224, 224, 3))
            
            # Get ground truth
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
            
            # Run inference
            pred_keypoints = run_hrnet_inference(hrnet_model, img_array, device)
            if pred_keypoints is None or pred_keypoints.shape[0] != 19:
                continue
            
            norm_preds_x.append(pred_keypoints[:, 0])
            norm_preds_y.append(pred_keypoints[:, 1])
            norm_gts_x.append(gt_keypoints[:, 0])
            norm_gts_y.append(gt_keypoints[:, 1])
            
        except Exception as e:
            continue
    
    if len(norm_preds_x) < 10:
        print("ERROR: Insufficient data for normalization")
        return
    
    # Create scalers
    norm_preds_x = np.array(norm_preds_x)
    norm_preds_y = np.array(norm_preds_y)
    norm_gts_x = np.array(norm_gts_x)
    norm_gts_y = np.array(norm_gts_y)
    
    scaler_x_input = StandardScaler()
    scaler_x_target = StandardScaler()
    scaler_y_input = StandardScaler()
    scaler_y_target = StandardScaler()
    
    scaler_x_input.fit(norm_preds_x)
    scaler_x_target.fit(norm_gts_x)
    scaler_y_input.fit(norm_preds_y)
    scaler_y_target.fit(norm_gts_y)
    
    print(f"✓ Normalization statistics computed from {len(norm_preds_x)} samples")
    
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
            
            # Run HRNetV2 inference
            pred_keypoints = run_hrnet_inference(hrnet_model, img_array, device)
            if pred_keypoints is None or pred_keypoints.shape[0] != 19:
                continue
            
            # Apply MLP refinement
            refined_x, refined_y = apply_mlp_refinement(
                pred_keypoints[:, 0], pred_keypoints[:, 1],
                mlp_x, mlp_y,
                scaler_x_input, scaler_x_target, scaler_y_input, scaler_y_target,
                device
            )
            
            # Store results
            hrnet_predictions.append(pred_keypoints)
            mlp_predictions.append(np.column_stack([refined_x, refined_y]))
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
    print("EVALUATION RESULTS")
    print("="*80)
    
    print(f"\n🏷️  OVERALL PERFORMANCE:")
    print(f"{'Metric':<15} {'HRNetV2':<15} {'MLP Refined':<15} {'Improvement':<15}")
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
    problematic_landmarks = ['sella', 'Gonion', 'PNS', 'A point', 'B point']
    
    print(f"{'Landmark':<20} {'HRNetV2 MRE':<15} {'MLP MRE':<15} {'Improvement':<15}")
    print("-" * 70)
    
    for landmark in problematic_landmarks:
        if landmark in hrnet_per_landmark and landmark in mlp_per_landmark:
            hrnet_err = hrnet_per_landmark[landmark]['mre']
            mlp_err = mlp_per_landmark[landmark]['mre']
            if hrnet_err > 0:
                improvement = (hrnet_err - mlp_err) / hrnet_err * 100
                print(f"{landmark:<20} {hrnet_err:<15.3f} {mlp_err:<15.3f} {improvement:<15.1f}%")
    
    # Save results
    output_dir = os.path.join(args.work_dir, "mlp_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_summary = {
        'hrnet_overall': hrnet_overall,
        'mlp_overall': mlp_overall,
        'improvement_mre': improvement_mre,
        'improvement_std': improvement_std,
        'improvement_median': improvement_median,
        'total_samples': len(hrnet_predictions)
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
    methods = ['HRNetV2', 'MLP Refined']
    mres = [hrnet_overall['mre'], mlp_overall['mre']]
    stds = [hrnet_overall['std'], mlp_overall['std']]
    
    ax1.bar(methods, mres, yerr=stds, capsize=5, alpha=0.7, color=['skyblue', 'lightcoral'])
    ax1.set_ylabel('Mean Radial Error (pixels)')
    ax1.set_title('Overall Performance Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Add improvement percentage
    ax1.text(1, mlp_overall['mre'] + mlp_overall['std'] + 0.1, 
             f'{improvement_mre:.1f}% improvement', ha='center', va='bottom', 
             fontsize=10, color='green', fontweight='bold')
    
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
             label=['HRNetV2', 'MLP Refined'], color=['skyblue', 'lightcoral'])
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
    ax4.set_ylabel('MLP Refined Error (pixels)')
    ax4.set_title('Error Correlation (Sample)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "mlp_evaluation_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n💾 Results saved to: {output_dir}")
    print(f"   - Per-landmark comparison: per_landmark_comparison.csv")
    print(f"   - Visualization: mlp_evaluation_results.png")
    
    print(f"\n🎉 Evaluation completed!")
    print(f"📈 Overall improvement: {improvement_mre:.1f}% reduction in MRE")
    print(f"🎯 Best performing landmarks benefit most from MLP refinement")

if __name__ == "__main__":
    main() 