#!/usr/bin/env python3
"""
Ensemble Concurrent Joint MLP Performance Evaluation Script
This script evaluates individual models and ensemble performance during training.
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
from typing import List, Dict, Tuple, Optional

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

def load_model_components(model_dir: str, device: torch.device, config_path: str) -> Optional[Tuple]:
    """Load HRNet model, MLP model, and scalers for a single ensemble model."""
    print(f"\n🔄 Loading model from: {os.path.basename(model_dir)}")
    
    # Find HRNet checkpoint
    hrnet_checkpoint_pattern = os.path.join(model_dir, "best_NME_epoch_*.pth")
    hrnet_checkpoints = glob.glob(hrnet_checkpoint_pattern)
    
    if not hrnet_checkpoints:
        hrnet_checkpoint_pattern = os.path.join(model_dir, "epoch_*.pth")
        hrnet_checkpoints = glob.glob(hrnet_checkpoint_pattern)
    
    if not hrnet_checkpoints:
        print(f"   ❌ No HRNet checkpoints found in {model_dir}")
        return None
    
    hrnet_checkpoint = max(hrnet_checkpoints, key=os.path.getctime)
    hrnet_checkpoint_name = os.path.basename(hrnet_checkpoint)
    print(f"   ✓ HRNet checkpoint: {hrnet_checkpoint_name}")
    
    # Load HRNet model
    try:
        hrnet_model = init_model(config_path, hrnet_checkpoint, device=device)
    except Exception as e:
        print(f"   ❌ Failed to load HRNet model: {e}")
        return None
    
    # Find MLP model and scalers
    mlp_dir = os.path.join(model_dir, "concurrent_mlp")
    if not os.path.exists(mlp_dir):
        print(f"   ❌ MLP directory not found: {mlp_dir}")
        return None
    
    # Load checkpoint mapping for synchronized model
    mapping_file = os.path.join(mlp_dir, "checkpoint_mlp_mapping.json")
    synchronized_mlp_path = None
    model_type = "unknown"
    
    if os.path.exists(mapping_file):
        try:
            import json
            with open(mapping_file, 'r') as f:
                checkpoint_mapping = json.load(f)
            
            if hrnet_checkpoint_name in checkpoint_mapping:
                synchronized_mlp_path = checkpoint_mapping[hrnet_checkpoint_name]
                if os.path.exists(synchronized_mlp_path):
                    print(f"   ✓ Synchronized MLP: {os.path.basename(synchronized_mlp_path)}")
                    model_type = f"synchronized"
                else:
                    synchronized_mlp_path = None
        except Exception as e:
            print(f"   ⚠️  Failed to load checkpoint mapping: {e}")
    
    # Fallback to epoch-based matching
    if synchronized_mlp_path is None:
        hrnet_epoch = None
        if "epoch_" in hrnet_checkpoint_name:
            try:
                hrnet_epoch = int(hrnet_checkpoint_name.split("epoch_")[1].split(".")[0])
            except:
                pass
        
        if hrnet_epoch is not None:
            epoch_mlp_path = os.path.join(mlp_dir, f"mlp_joint_epoch_{hrnet_epoch}.pth")
            if os.path.exists(epoch_mlp_path):
                synchronized_mlp_path = epoch_mlp_path
                model_type = f"epoch_{hrnet_epoch}"
                print(f"   ✓ Epoch-matched MLP: {os.path.basename(epoch_mlp_path)}")
        
        # Final fallbacks
        if synchronized_mlp_path is None:
            fallback_paths = [
                os.path.join(mlp_dir, "mlp_joint_final.pth"),
                os.path.join(mlp_dir, "mlp_joint_latest.pth")
            ]
            
            for path in fallback_paths:
                if os.path.exists(path):
                    synchronized_mlp_path = path
                    model_type = f"fallback_{os.path.basename(path)}"
                    print(f"   ✓ Fallback MLP: {os.path.basename(path)}")
                    break
            
            if synchronized_mlp_path is None:
                # Try any epoch model
                epoch_models = glob.glob(os.path.join(mlp_dir, "mlp_joint_epoch_*.pth"))
                if epoch_models:
                    synchronized_mlp_path = max(epoch_models, key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
                    epoch_num = synchronized_mlp_path.split('_epoch_')[1].split('.')[0]
                    model_type = f"latest_epoch_{epoch_num}"
                    print(f"   ✓ Latest epoch MLP: {os.path.basename(synchronized_mlp_path)}")
                else:
                    print(f"   ❌ No MLP models found")
                    return None
    
    # Load MLP model
    try:
        mlp_joint = JointMLPRefinementModel().to(device)
        mlp_joint.load_state_dict(torch.load(synchronized_mlp_path, map_location=device))
        mlp_joint.eval()
    except Exception as e:
        print(f"   ❌ Failed to load MLP model: {e}")
        return None
    
    # Load scalers
    scaler_input_path = os.path.join(mlp_dir, "scaler_joint_input.pkl")
    scaler_target_path = os.path.join(mlp_dir, "scaler_joint_target.pkl")
    
    if not os.path.exists(scaler_input_path) or not os.path.exists(scaler_target_path):
        print(f"   ❌ Scalers not found")
        return None
    
    try:
        scaler_input = joblib.load(scaler_input_path)
        scaler_target = joblib.load(scaler_target_path)
        print(f"   ✓ Scalers loaded")
    except Exception as e:
        print(f"   ❌ Failed to load scalers: {e}")
        return None
    
    return hrnet_model, mlp_joint, scaler_input, scaler_target, model_type, hrnet_checkpoint_name

def evaluate_single_model(hrnet_model, mlp_joint, scaler_input, scaler_target, 
                         test_df, landmark_names, landmark_cols, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a single model and return predictions."""
    hrnet_predictions = []
    mlp_predictions = []
    ground_truths = []
    
    print(f"   🔄 Running inference on {len(test_df)} samples...")
    
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
            
            if not valid_gt:
                continue
                
            gt_keypoints = np.array(gt_keypoints)
            
            # Run HRNetV2 inference
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
            continue
    
    if len(hrnet_predictions) == 0:
        return None, None, None
    
    return np.array(hrnet_predictions), np.array(mlp_predictions), np.array(ground_truths)

def compute_metrics(pred_coords, gt_coords, landmark_names) -> Tuple[Dict, Dict]:
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
        'max': np.max(valid_errors),
        'count': len(valid_errors)
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

def create_ensemble_predictions(all_hrnet_preds: List[np.ndarray], 
                              all_mlp_preds: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Create ensemble predictions by averaging individual model predictions."""
    print(f"\n🔄 Creating ensemble predictions from {len(all_hrnet_preds)} models...")
    
    # Average HRNet predictions
    ensemble_hrnet = np.mean(all_hrnet_preds, axis=0)
    
    # Average MLP predictions  
    ensemble_mlp = np.mean(all_mlp_preds, axis=0)
    
    print(f"   ✓ Ensemble shape: {ensemble_hrnet.shape}")
    
    return ensemble_hrnet, ensemble_mlp

def print_results_table(results: Dict[str, Dict], landmark_names: List[str]):
    """Print formatted results table."""
    print(f"\n{'='*100}")
    print(f"📊 ENSEMBLE EVALUATION RESULTS")
    print(f"{'='*100}")
    
    # Overall performance table
    print(f"\n🏷️  OVERALL PERFORMANCE:")
    header = f"{'Model':<20} {'MRE':<10} {'Std':<10} {'Median':<10} {'P90':<10} {'P95':<10} {'Samples':<10}"
    print(header)
    print("-" * len(header))
    
    for model_name, metrics in results.items():
        overall = metrics['overall']
        print(f"{model_name:<20} {overall['mre']:<10.3f} {overall['std']:<10.3f} "
              f"{overall['median']:<10.3f} {overall['p90']:<10.3f} {overall['p95']:<10.3f} "
              f"{overall['count']:<10}")
    
    # Key landmarks performance
    key_landmarks = ['sella', 'Gonion', 'PNS', 'A_point', 'B_point', 'ANS', 'nasion']
    available_landmarks = [lm for lm in key_landmarks if lm in landmark_names]
    
    print(f"\n🎯 KEY LANDMARKS PERFORMANCE:")
    
    for landmark in available_landmarks:
        print(f"\n{landmark.upper()}:")
        header = f"{'Model':<20} {'MRE':<10} {'Std':<10} {'Median':<10} {'Count':<10}"
        print(header)
        print("-" * len(header))
        
        for model_name, metrics in results.items():
            if landmark in metrics['per_landmark']:
                lm_metrics = metrics['per_landmark'][landmark]
                print(f"{model_name:<20} {lm_metrics['mre']:<10.3f} {lm_metrics['std']:<10.3f} "
                      f"{lm_metrics['median']:<10.3f} {lm_metrics['count']:<10}")
    
    # Improvement analysis (compare with first individual model)
    if len(results) > 2:  # At least 2 individual models + ensemble
        individual_models = [k for k in results.keys() if k.startswith('Model')]
        if individual_models and 'Ensemble MLP' in results:
            baseline_model = individual_models[0] + ' MLP'
            if baseline_model in results:
                baseline_mre = results[baseline_model]['overall']['mre']
                ensemble_mre = results['Ensemble MLP']['overall']['mre']
                
                improvement = (baseline_mre - ensemble_mre) / baseline_mre * 100
                
                print(f"\n📈 ENSEMBLE IMPROVEMENT:")
                print(f"   Baseline ({baseline_model}): {baseline_mre:.3f} pixels")
                print(f"   Ensemble MLP: {ensemble_mre:.3f} pixels")
                print(f"   Improvement: {improvement:+.1f}%")

def main():
    """Main ensemble evaluation function."""
    
    parser = argparse.ArgumentParser(
        description='Evaluate Ensemble Concurrent Joint MLP Performance')
    parser.add_argument(
        '--test_split_file',
        type=str,
        default=None,
        help='Path to a text file containing patient IDs for the test set, one ID per line.'
    )
    parser.add_argument(
        '--base_work_dir',
        type=str,
        default='work_dirs/hrnetv2_w18_cephalometric_ensemble_concurrent_mlp_v5',
        help='Base work directory containing the ensemble models'
    )
    parser.add_argument(
        '--n_models',
        type=int,
        default=3,
        help='Number of models in the ensemble (default: 3)'
    )
    parser.add_argument(
        '--evaluate_individual',
        action='store_true',
        help='Evaluate individual models separately'
    )
    args = parser.parse_args()
    
    print("="*80)
    print("ENSEMBLE CONCURRENT JOINT MLP EVALUATION")
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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
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
    
    print(f"✓ Evaluating on {len(test_df)} test samples")
    
    # Get landmark information
    landmark_names = cephalometric_dataset_info.landmark_names_in_order
    landmark_cols = cephalometric_dataset_info.original_landmark_cols
    
    # Find model directories
    model_dirs = []
    for i in range(1, args.n_models + 1):
        model_dir = os.path.join(args.base_work_dir, f"model_{i}")
        if os.path.exists(model_dir):
            model_dirs.append(model_dir)
        else:
            print(f"⚠️  Model directory not found: {model_dir}")
    
    if not model_dirs:
        print("ERROR: No model directories found")
        return
    
    print(f"✓ Found {len(model_dirs)} model directories")
    
    # Load and evaluate models
    all_model_components = []
    all_hrnet_preds = []
    all_mlp_preds = []
    all_gt = None
    results = {}
    
    for i, model_dir in enumerate(model_dirs, 1):
        components = load_model_components(model_dir, device, config_path)
        
        if components is None:
            print(f"   ❌ Skipping model {i} due to loading errors")
            continue
        
        hrnet_model, mlp_joint, scaler_input, scaler_target, model_type, checkpoint_name = components
        all_model_components.append((hrnet_model, mlp_joint, scaler_input, scaler_target, model_type, checkpoint_name))
        
        # Evaluate this model
        hrnet_preds, mlp_preds, gt_coords = evaluate_single_model(
            hrnet_model, mlp_joint, scaler_input, scaler_target,
            test_df, landmark_names, landmark_cols, device
        )
        
        if hrnet_preds is None:
            print(f"   ❌ No valid predictions from model {i}")
            continue
        
        print(f"   ✓ Model {i} evaluated: {len(hrnet_preds)} samples")
        
        all_hrnet_preds.append(hrnet_preds)
        all_mlp_preds.append(mlp_preds)
        
        if all_gt is None:
            all_gt = gt_coords
        
        # Compute metrics for individual model if requested
        if args.evaluate_individual:
            hrnet_overall, hrnet_per_landmark = compute_metrics(hrnet_preds, gt_coords, landmark_names)
            mlp_overall, mlp_per_landmark = compute_metrics(mlp_preds, gt_coords, landmark_names)
            
            results[f'Model {i} HRNet'] = {'overall': hrnet_overall, 'per_landmark': hrnet_per_landmark}
            results[f'Model {i} MLP'] = {'overall': mlp_overall, 'per_landmark': mlp_per_landmark}
    
    if not all_hrnet_preds:
        print("ERROR: No models successfully evaluated")
        return
    
    # Create ensemble predictions
    ensemble_hrnet, ensemble_mlp = create_ensemble_predictions(all_hrnet_preds, all_mlp_preds)
    
    # Evaluate ensemble
    print(f"\n🔄 Computing ensemble metrics...")
    ensemble_hrnet_overall, ensemble_hrnet_per_landmark = compute_metrics(ensemble_hrnet, all_gt, landmark_names)
    ensemble_mlp_overall, ensemble_mlp_per_landmark = compute_metrics(ensemble_mlp, all_gt, landmark_names)
    
    results['Ensemble HRNet'] = {'overall': ensemble_hrnet_overall, 'per_landmark': ensemble_hrnet_per_landmark}
    results['Ensemble MLP'] = {'overall': ensemble_mlp_overall, 'per_landmark': ensemble_mlp_per_landmark}
    
    # Print results
    print_results_table(results, landmark_names)
    
    # Save results
    output_dir = os.path.join(args.base_work_dir, "ensemble_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed comparison
    comparison_data = []
    for result_name, metrics in results.items():
        comparison_data.append({
            'model': result_name,
            'mre': metrics['overall']['mre'],
            'std': metrics['overall']['std'],
            'median': metrics['overall']['median'],
            'p90': metrics['overall']['p90'],
            'p95': metrics['overall']['p95'],
            'samples': metrics['overall']['count']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(output_dir, "ensemble_comparison.csv"), index=False)
    
    # Save per-landmark results for key landmarks
    key_landmarks = ['sella', 'Gonion', 'PNS', 'A_point', 'B_point', 'ANS', 'nasion']
    landmark_data = []
    
    for landmark in key_landmarks:
        if landmark in landmark_names:
            for result_name, metrics in results.items():
                if landmark in metrics['per_landmark']:
                    lm_metrics = metrics['per_landmark'][landmark]
                    landmark_data.append({
                        'model': result_name,
                        'landmark': landmark,
                        'mre': lm_metrics['mre'],
                        'std': lm_metrics['std'],
                        'median': lm_metrics['median'],
                        'count': lm_metrics['count']
                    })
    
    landmark_df = pd.DataFrame(landmark_data)
    landmark_df.to_csv(os.path.join(output_dir, "key_landmarks_comparison.csv"), index=False)
    
    print(f"\n💾 Results saved to: {output_dir}")
    print(f"   - Overall comparison: ensemble_comparison.csv")
    print(f"   - Key landmarks: key_landmarks_comparison.csv")
    
    # Quick summary
    ensemble_mre = ensemble_mlp_overall['mre']
    print(f"\n🎉 Ensemble Evaluation Summary:")
    print(f"📊 {len(all_hrnet_preds)} models successfully evaluated")
    print(f"🎯 Ensemble MLP MRE: {ensemble_mre:.3f} pixels")
    
    if len(all_hrnet_preds) > 1:
        individual_mres = [results[f'Model {i+1} MLP']['overall']['mre'] for i in range(len(all_hrnet_preds))]
        avg_individual_mre = np.mean(individual_mres)
        ensemble_improvement = (avg_individual_mre - ensemble_mre) / avg_individual_mre * 100
        print(f"📈 Improvement over average individual: {ensemble_improvement:+.1f}%")
    
    # Sella-specific summary
    if 'sella' in landmark_names:
        sella_results = [(name, metrics['per_landmark'].get('sella', {}).get('mre', 0)) 
                        for name, metrics in results.items() if 'MLP' in name]
        
        print(f"\n🎯 SELLA LANDMARK PERFORMANCE:")
        for model_name, sella_mre in sella_results:
            print(f"   {model_name}: {sella_mre:.3f} pixels")

if __name__ == "__main__":
    main() 