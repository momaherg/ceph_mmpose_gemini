#!/usr/bin/env python3
"""
Enhanced Ensemble Concurrent Joint MLP Performance Evaluation Script
This script evaluates individual models and ensemble performance during training,
and saves all predictions with patient IDs and relevant information.
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
import json
from datetime import datetime

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

def evaluate_single_model_with_tracking(hrnet_model, mlp_joint, scaler_input, scaler_target, 
                                       test_df, landmark_names, landmark_cols, device, model_idx) -> Dict:
    """Evaluate a single model and return predictions with patient tracking."""
    predictions_data = []
    
    print(f"   🔄 Running inference on {len(test_df)} samples for Model {model_idx}...")
    
    for idx, row in test_df.iterrows():
        try:
            # Get patient information
            patient_id = row.get('patient_id', -1)
            
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
            
            # Store results for this patient
            patient_data = {
                'patient_id': int(patient_id),
                'dataframe_index': idx,
                'ground_truth': gt_keypoints.tolist(),
                f'model_{model_idx}_hrnet': pred_keypoints.tolist(),
                f'model_{model_idx}_mlp': refined_keypoints.tolist()
            }
            
            predictions_data.append(patient_data)
            
        except Exception as e:
            print(f"   ⚠️  Failed to process sample {idx}: {e}")
            continue
    
    print(f"   ✓ Model {model_idx} processed {len(predictions_data)} valid samples")
    
    return predictions_data

def merge_predictions_data(all_predictions_data: List[List[Dict]], landmark_names: List[str]) -> pd.DataFrame:
    """Merge predictions from all models into a single DataFrame."""
    # Create a dictionary to store all predictions by patient_id
    merged_data = {}
    
    # First pass: collect all unique patient IDs and ground truth
    for model_idx, model_predictions in enumerate(all_predictions_data, 1):
        for patient_data in model_predictions:
            patient_id = patient_data['patient_id']
            
            if patient_id not in merged_data:
                merged_data[patient_id] = {
                    'patient_id': patient_id,
                    'dataframe_index': patient_data['dataframe_index'],
                    'ground_truth': patient_data['ground_truth']
                }
            
            # Add model predictions
            merged_data[patient_id][f'model_{model_idx}_hrnet'] = patient_data[f'model_{model_idx}_hrnet']
            merged_data[patient_id][f'model_{model_idx}_mlp'] = patient_data[f'model_{model_idx}_mlp']
    
    # Convert to list for DataFrame creation
    final_data = []
    
    for patient_id, patient_data in merged_data.items():
        row_data = {
            'patient_id': patient_id,
            'dataframe_index': patient_data['dataframe_index']
        }
        
        # Add ground truth coordinates
        gt = np.array(patient_data['ground_truth'])
        for i, landmark_name in enumerate(landmark_names):
            row_data[f'gt_{landmark_name}_x'] = gt[i, 0]
            row_data[f'gt_{landmark_name}_y'] = gt[i, 1]
        
        # Add individual model predictions
        n_models = len(all_predictions_data)
        
        # Collect all model predictions for ensemble calculation
        all_hrnet_preds = []
        all_mlp_preds = []
        
        for model_idx in range(1, n_models + 1):
            hrnet_key = f'model_{model_idx}_hrnet'
            mlp_key = f'model_{model_idx}_mlp'
            
            if hrnet_key in patient_data:
                hrnet_pred = np.array(patient_data[hrnet_key])
                mlp_pred = np.array(patient_data[mlp_key])
                
                all_hrnet_preds.append(hrnet_pred)
                all_mlp_preds.append(mlp_pred)
                
                # Add individual model predictions
                for i, landmark_name in enumerate(landmark_names):
                    row_data[f'model_{model_idx}_hrnet_{landmark_name}_x'] = hrnet_pred[i, 0]
                    row_data[f'model_{model_idx}_hrnet_{landmark_name}_y'] = hrnet_pred[i, 1]
                    row_data[f'model_{model_idx}_mlp_{landmark_name}_x'] = mlp_pred[i, 0]
                    row_data[f'model_{model_idx}_mlp_{landmark_name}_y'] = mlp_pred[i, 1]
        
        # Calculate ensemble predictions if we have predictions from all models
        if len(all_hrnet_preds) == n_models:
            ensemble_hrnet = np.mean(all_hrnet_preds, axis=0)
            ensemble_mlp = np.mean(all_mlp_preds, axis=0)
            
            # Add ensemble predictions
            for i, landmark_name in enumerate(landmark_names):
                row_data[f'ensemble_hrnet_{landmark_name}_x'] = ensemble_hrnet[i, 0]
                row_data[f'ensemble_hrnet_{landmark_name}_y'] = ensemble_hrnet[i, 1]
                row_data[f'ensemble_mlp_{landmark_name}_x'] = ensemble_mlp[i, 0]
                row_data[f'ensemble_mlp_{landmark_name}_y'] = ensemble_mlp[i, 1]
                
                # Calculate errors
                gt_x = gt[i, 0]
                gt_y = gt[i, 1]
                
                if gt_x > 0 and gt_y > 0:  # Valid ground truth
                    # Individual model errors
                    for model_idx in range(1, n_models + 1):
                        hrnet_x = row_data[f'model_{model_idx}_hrnet_{landmark_name}_x']
                        hrnet_y = row_data[f'model_{model_idx}_hrnet_{landmark_name}_y']
                        mlp_x = row_data[f'model_{model_idx}_mlp_{landmark_name}_x']
                        mlp_y = row_data[f'model_{model_idx}_mlp_{landmark_name}_y']
                        
                        hrnet_error = np.sqrt((hrnet_x - gt_x)**2 + (hrnet_y - gt_y)**2)
                        mlp_error = np.sqrt((mlp_x - gt_x)**2 + (mlp_y - gt_y)**2)
                        
                        row_data[f'model_{model_idx}_hrnet_{landmark_name}_error'] = hrnet_error
                        row_data[f'model_{model_idx}_mlp_{landmark_name}_error'] = mlp_error
                    
                    # Ensemble errors
                    ensemble_hrnet_error = np.sqrt((ensemble_hrnet[i, 0] - gt_x)**2 + (ensemble_hrnet[i, 1] - gt_y)**2)
                    ensemble_mlp_error = np.sqrt((ensemble_mlp[i, 0] - gt_x)**2 + (ensemble_mlp[i, 1] - gt_y)**2)
                    
                    row_data[f'ensemble_hrnet_{landmark_name}_error'] = ensemble_hrnet_error
                    row_data[f'ensemble_mlp_{landmark_name}_error'] = ensemble_mlp_error
        
        final_data.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(final_data)
    
    # Sort by patient_id
    df = df.sort_values('patient_id').reset_index(drop=True)
    
    return df

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

def main():
    """Main ensemble evaluation function with prediction saving."""
    
    parser = argparse.ArgumentParser(
        description='Evaluate Ensemble Concurrent Joint MLP Performance and Save Predictions')
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
        '--save_format',
        type=str,
        choices=['csv', 'json', 'both'],
        default='both',
        help='Format to save predictions (default: both)'
    )
    args = parser.parse_args()
    
    print("="*80)
    print("ENSEMBLE EVALUATION WITH PREDICTION SAVING")
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
    
    # Load and evaluate models with prediction tracking
    all_predictions_data = []
    
    for i, model_dir in enumerate(model_dirs, 1):
        components = load_model_components(model_dir, device, config_path)
        
        if components is None:
            print(f"   ❌ Skipping model {i} due to loading errors")
            continue
        
        hrnet_model, mlp_joint, scaler_input, scaler_target, model_type, checkpoint_name = components
        
        # Evaluate this model on test set with patient tracking
        predictions_data = evaluate_single_model_with_tracking(
            hrnet_model, mlp_joint, scaler_input, scaler_target,
            test_df, landmark_names, landmark_cols, device, i
        )
        
        if not predictions_data:
            print(f"   ❌ No valid predictions from model {i}")
            continue
        
        all_predictions_data.append(predictions_data)
    
    if not all_predictions_data:
        print("ERROR: No models successfully evaluated")
        return
    
    # Merge all predictions into a single DataFrame
    print(f"\n🔄 Merging predictions from {len(all_predictions_data)} models...")
    predictions_df = merge_predictions_data(all_predictions_data, landmark_names)
    print(f"✓ Merged predictions for {len(predictions_df)} patients")
    
    # Save results
    output_dir = os.path.join(args.base_work_dir, "ensemble_predictions")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as CSV
    if args.save_format in ['csv', 'both']:
        csv_path = os.path.join(output_dir, f"ensemble_predictions_{timestamp}.csv")
        predictions_df.to_csv(csv_path, index=False)
        print(f"✓ Predictions saved to CSV: {csv_path}")
        
        # Also save a more compact version with just ensemble predictions
        compact_columns = ['patient_id', 'dataframe_index']
        
        # Add ground truth
        for landmark in landmark_names:
            compact_columns.extend([f'gt_{landmark}_x', f'gt_{landmark}_y'])
        
        # Add ensemble predictions and errors
        for landmark in landmark_names:
            compact_columns.extend([
                f'ensemble_mlp_{landmark}_x',
                f'ensemble_mlp_{landmark}_y',
                f'ensemble_mlp_{landmark}_error'
            ])
        
        # Filter columns that exist
        compact_columns = [col for col in compact_columns if col in predictions_df.columns]
        compact_df = predictions_df[compact_columns]
        
        compact_csv_path = os.path.join(output_dir, f"ensemble_predictions_compact_{timestamp}.csv")
        compact_df.to_csv(compact_csv_path, index=False)
        print(f"✓ Compact predictions saved to CSV: {compact_csv_path}")
    
    # Save as JSON
    if args.save_format in ['json', 'both']:
        # Create structured JSON format
        json_data = {
            'metadata': {
                'timestamp': timestamp,
                'n_models': len(all_predictions_data),
                'n_patients': len(predictions_df),
                'landmark_names': landmark_names,
                'model_directories': model_dirs
            },
            'predictions': []
        }
        
        for _, row in predictions_df.iterrows():
            patient_data = {
                'patient_id': int(row['patient_id']),
                'dataframe_index': int(row['dataframe_index']),
                'ground_truth': {},
                'individual_models': {},
                'ensemble': {}
            }
            
            # Add ground truth
            for landmark in landmark_names:
                patient_data['ground_truth'][landmark] = {
                    'x': float(row[f'gt_{landmark}_x']),
                    'y': float(row[f'gt_{landmark}_y'])
                }
            
            # Add individual model predictions
            n_models = len(all_predictions_data)
            for model_idx in range(1, n_models + 1):
                patient_data['individual_models'][f'model_{model_idx}'] = {
                    'hrnet': {},
                    'mlp': {}
                }
                
                for landmark in landmark_names:
                    if f'model_{model_idx}_hrnet_{landmark}_x' in row:
                        patient_data['individual_models'][f'model_{model_idx}']['hrnet'][landmark] = {
                            'x': float(row[f'model_{model_idx}_hrnet_{landmark}_x']),
                            'y': float(row[f'model_{model_idx}_hrnet_{landmark}_y']),
                            'error': float(row.get(f'model_{model_idx}_hrnet_{landmark}_error', -1))
                        }
                        patient_data['individual_models'][f'model_{model_idx}']['mlp'][landmark] = {
                            'x': float(row[f'model_{model_idx}_mlp_{landmark}_x']),
                            'y': float(row[f'model_{model_idx}_mlp_{landmark}_y']),
                            'error': float(row.get(f'model_{model_idx}_mlp_{landmark}_error', -1))
                        }
            
            # Add ensemble predictions
            for landmark in landmark_names:
                if f'ensemble_hrnet_{landmark}_x' in row:
                    patient_data['ensemble']['hrnet'] = patient_data['ensemble'].get('hrnet', {})
                    patient_data['ensemble']['hrnet'][landmark] = {
                        'x': float(row[f'ensemble_hrnet_{landmark}_x']),
                        'y': float(row[f'ensemble_hrnet_{landmark}_y']),
                        'error': float(row.get(f'ensemble_hrnet_{landmark}_error', -1))
                    }
                    
                    patient_data['ensemble']['mlp'] = patient_data['ensemble'].get('mlp', {})
                    patient_data['ensemble']['mlp'][landmark] = {
                        'x': float(row[f'ensemble_mlp_{landmark}_x']),
                        'y': float(row[f'ensemble_mlp_{landmark}_y']),
                        'error': float(row.get(f'ensemble_mlp_{landmark}_error', -1))
                    }
            
            json_data['predictions'].append(patient_data)
        
        json_path = os.path.join(output_dir, f"ensemble_predictions_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"✓ Predictions saved to JSON: {json_path}")
    
    # Compute and display metrics
    print(f"\n📊 COMPUTING METRICS...")
    
    # Extract arrays for metric computation
    n_patients = len(predictions_df)
    n_landmarks = len(landmark_names)
    
    gt_coords = np.zeros((n_patients, n_landmarks, 2))
    ensemble_mlp_coords = np.zeros((n_patients, n_landmarks, 2))
    
    for i, (_, row) in enumerate(predictions_df.iterrows()):
        for j, landmark in enumerate(landmark_names):
            gt_coords[i, j, 0] = row[f'gt_{landmark}_x']
            gt_coords[i, j, 1] = row[f'gt_{landmark}_y']
            
            if f'ensemble_mlp_{landmark}_x' in row:
                ensemble_mlp_coords[i, j, 0] = row[f'ensemble_mlp_{landmark}_x']
                ensemble_mlp_coords[i, j, 1] = row[f'ensemble_mlp_{landmark}_y']
    
    # Compute metrics
    overall_metrics, per_landmark_metrics = compute_metrics(ensemble_mlp_coords, gt_coords, landmark_names)
    
    print(f"\n🎯 ENSEMBLE MLP PERFORMANCE:")
    print(f"   Mean Radial Error: {overall_metrics['mre']:.3f} pixels")
    print(f"   Std Dev: {overall_metrics['std']:.3f} pixels")
    print(f"   Median: {overall_metrics['median']:.3f} pixels")
    print(f"   P90: {overall_metrics['p90']:.3f} pixels")
    print(f"   P95: {overall_metrics['p95']:.3f} pixels")
    
    # Key landmarks
    print(f"\n🎯 KEY LANDMARKS:")
    key_landmarks = ['sella', 'Gonion', 'PNS', 'A_point', 'B_point']
    for landmark in key_landmarks:
        if landmark in per_landmark_metrics:
            metrics = per_landmark_metrics[landmark]
            print(f"   {landmark}: MRE={metrics['mre']:.3f}, Std={metrics['std']:.3f}")
    
    print(f"\n🎉 Evaluation completed successfully!")
    print(f"📁 All predictions saved to: {output_dir}")
    print(f"📋 Files created:")
    if args.save_format in ['csv', 'both']:
        print(f"   - Full predictions CSV: ensemble_predictions_{timestamp}.csv")
        print(f"   - Compact predictions CSV: ensemble_predictions_compact_{timestamp}.csv")
    if args.save_format in ['json', 'both']:
        print(f"   - Structured JSON: ensemble_predictions_{timestamp}.json")

if __name__ == "__main__":
    main() 