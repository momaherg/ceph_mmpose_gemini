#!/usr/bin/env python3
"""
Save Ensemble Predictions with Patient IDs
This script evaluates ensemble models and saves predictions with patient IDs and errors.
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
        # Try to find MLP model
        mlp_paths = [
            os.path.join(mlp_dir, "mlp_joint_final.pth"),
            os.path.join(mlp_dir, "mlp_joint_latest.pth")
        ]
        
        for path in mlp_paths:
            if os.path.exists(path):
                synchronized_mlp_path = path
                print(f"   ✓ Found MLP: {os.path.basename(path)}")
                break
        
        if synchronized_mlp_path is None:
            # Try any epoch model
            epoch_models = glob.glob(os.path.join(mlp_dir, "mlp_joint_epoch_*.pth"))
            if epoch_models:
                synchronized_mlp_path = max(epoch_models, key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
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

def evaluate_single_model_with_ids(hrnet_model, mlp_joint, scaler_input, scaler_target, 
                                 test_df, landmark_names, landmark_cols, device) -> Tuple:
    """Evaluate a single model and return predictions with patient IDs."""
    hrnet_predictions = []
    mlp_predictions = []
    ground_truths = []
    patient_ids = []
    
    print(f"   🔄 Running inference on {len(test_df)} samples...")
    
    for idx, row in test_df.iterrows():
        try:
            # Get patient ID
            patient_id = row.get('patient_id', idx)
            
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
            patient_ids.append(patient_id)
            
        except Exception as e:
            continue
    
    if len(hrnet_predictions) == 0:
        return None, None, None, None
    
    return (np.array(hrnet_predictions), np.array(mlp_predictions), 
            np.array(ground_truths), np.array(patient_ids))

def save_ensemble_predictions(ensemble_hrnet, ensemble_mlp, ground_truths, patient_ids, 
                            landmark_names, output_dir):
    """Save ensemble predictions to CSV with patient IDs and errors."""
    print("\n💾 Saving ensemble predictions...")
    
    # Prepare data for CSV
    rows = []
    
    for i, patient_id in enumerate(patient_ids):
        row = {'patient_id': patient_id}
        
        # Add ground truth and predictions for each landmark
        for j, landmark in enumerate(landmark_names):
            # Ground truth
            row[f'gt_{landmark}_x'] = ground_truths[i, j, 0]
            row[f'gt_{landmark}_y'] = ground_truths[i, j, 1]
            
            # Ensemble MLP predictions
            row[f'ensemble_mlp_{landmark}_x'] = ensemble_mlp[i, j, 0]
            row[f'ensemble_mlp_{landmark}_y'] = ensemble_mlp[i, j, 1]
            
            # Calculate MLP error
            mlp_error = np.sqrt((ensemble_mlp[i, j, 0] - ground_truths[i, j, 0])**2 + 
                               (ensemble_mlp[i, j, 1] - ground_truths[i, j, 1])**2)
            row[f'ensemble_mlp_{landmark}_error'] = mlp_error
            
            # Ensemble HRNet predictions
            row[f'ensemble_hrnet_{landmark}_x'] = ensemble_hrnet[i, j, 0]
            row[f'ensemble_hrnet_{landmark}_y'] = ensemble_hrnet[i, j, 1]
            
            # Calculate HRNet error
            hrnet_error = np.sqrt((ensemble_hrnet[i, j, 0] - ground_truths[i, j, 0])**2 + 
                                 (ensemble_hrnet[i, j, 1] - ground_truths[i, j, 1])**2)
            row[f'ensemble_hrnet_{landmark}_error'] = hrnet_error
        
        rows.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'ensemble_predictions.csv')
    df.to_csv(csv_path, index=False)
    print(f"   ✓ Saved ensemble predictions to: {csv_path}")
    
    return df

def save_individual_model_predictions(all_hrnet_preds, all_mlp_preds, ground_truths, 
                                    patient_ids, landmark_names, output_dir, n_models):
    """Save individual model predictions to CSV with patient IDs and errors."""
    print("\n💾 Saving individual model predictions...")
    
    # Prepare data for CSV
    rows = []
    
    for i, patient_id in enumerate(patient_ids):
        row = {'patient_id': patient_id}
        
        # Add ground truth
        for j, landmark in enumerate(landmark_names):
            row[f'gt_{landmark}_x'] = ground_truths[i, j, 0]
            row[f'gt_{landmark}_y'] = ground_truths[i, j, 1]
        
        # Add predictions from each model
        for model_idx in range(n_models):
            if model_idx < len(all_hrnet_preds):
                hrnet_preds = all_hrnet_preds[model_idx]
                mlp_preds = all_mlp_preds[model_idx]
                
                for j, landmark in enumerate(landmark_names):
                    # Model HRNet predictions
                    row[f'model_{model_idx+1}_hrnet_{landmark}_x'] = hrnet_preds[i, j, 0]
                    row[f'model_{model_idx+1}_hrnet_{landmark}_y'] = hrnet_preds[i, j, 1]
                    
                    # Calculate HRNet error
                    hrnet_error = np.sqrt((hrnet_preds[i, j, 0] - ground_truths[i, j, 0])**2 + 
                                        (hrnet_preds[i, j, 1] - ground_truths[i, j, 1])**2)
                    row[f'model_{model_idx+1}_hrnet_{landmark}_error'] = hrnet_error
                    
                    # Model MLP predictions
                    row[f'model_{model_idx+1}_mlp_{landmark}_x'] = mlp_preds[i, j, 0]
                    row[f'model_{model_idx+1}_mlp_{landmark}_y'] = mlp_preds[i, j, 1]
                    
                    # Calculate MLP error
                    mlp_error = np.sqrt((mlp_preds[i, j, 0] - ground_truths[i, j, 0])**2 + 
                                       (mlp_preds[i, j, 1] - ground_truths[i, j, 1])**2)
                    row[f'model_{model_idx+1}_mlp_{landmark}_error'] = mlp_error
        
        rows.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'individual_model_predictions.csv')
    df.to_csv(csv_path, index=False)
    print(f"   ✓ Saved individual model predictions to: {csv_path}")
    
    return df

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

def main():
    """Main function to save ensemble predictions."""
    
    parser = argparse.ArgumentParser(
        description='Save Ensemble Predictions with Patient IDs')
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
    args = parser.parse_args()
    
    print("="*80)
    print("SAVE ENSEMBLE PREDICTIONS WITH PATIENT IDS")
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
    all_hrnet_preds = []
    all_mlp_preds = []
    all_gt = None
    all_patient_ids = None
    
    for i, model_dir in enumerate(model_dirs, 1):
        components = load_model_components(model_dir, device, config_path)
        
        if components is None:
            print(f"   ❌ Skipping model {i} due to loading errors")
            continue
        
        hrnet_model, mlp_joint, scaler_input, scaler_target, model_type, checkpoint_name = components
        
        # Evaluate this model on test set with patient IDs
        hrnet_preds, mlp_preds, gt_coords, patient_ids = evaluate_single_model_with_ids(
            hrnet_model, mlp_joint, scaler_input, scaler_target,
            test_df, landmark_names, landmark_cols, device
        )
        
        if hrnet_preds is None:
            print(f"   ❌ No valid predictions from model {i}")
            continue
        
        print(f"   ✓ Model {i} evaluated on test set: {len(hrnet_preds)} samples")
        
        all_hrnet_preds.append(hrnet_preds)
        all_mlp_preds.append(mlp_preds)
        
        if all_gt is None:
            all_gt = gt_coords
            all_patient_ids = patient_ids
    
    if not all_hrnet_preds:
        print("ERROR: No models successfully evaluated")
        return
    
    # Create ensemble predictions
    ensemble_hrnet, ensemble_mlp = create_ensemble_predictions(all_hrnet_preds, all_mlp_preds)
    
    # Create output directory
    output_dir = os.path.join(args.base_work_dir, "predictions")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save ensemble predictions
    ensemble_df = save_ensemble_predictions(
        ensemble_hrnet, ensemble_mlp, all_gt, all_patient_ids, 
        landmark_names, output_dir
    )
    
    # Save individual model predictions
    individual_df = save_individual_model_predictions(
        all_hrnet_preds, all_mlp_preds, all_gt, all_patient_ids,
        landmark_names, output_dir, args.n_models
    )
    
    # Print summary statistics
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Total patients evaluated: {len(all_patient_ids)}")
    print(f"   Models used in ensemble: {len(all_hrnet_preds)}")
    print(f"   Landmarks predicted: {len(landmark_names)}")
    
    # Calculate overall MRE for ensemble
    ensemble_mlp_errors = []
    ensemble_hrnet_errors = []
    
    for landmark in landmark_names:
        if f'ensemble_mlp_{landmark}_error' in ensemble_df.columns:
            ensemble_mlp_errors.extend(ensemble_df[f'ensemble_mlp_{landmark}_error'].values)
        if f'ensemble_hrnet_{landmark}_error' in ensemble_df.columns:
            ensemble_hrnet_errors.extend(ensemble_df[f'ensemble_hrnet_{landmark}_error'].values)
    
    if ensemble_mlp_errors:
        print(f"\n   Ensemble MLP Mean Radial Error: {np.mean(ensemble_mlp_errors):.3f} pixels")
    if ensemble_hrnet_errors:
        print(f"   Ensemble HRNet Mean Radial Error: {np.mean(ensemble_hrnet_errors):.3f} pixels")
    
    print(f"\n✅ Predictions saved successfully!")
    print(f"   📁 Output directory: {output_dir}")
    print(f"   📄 Ensemble predictions: ensemble_predictions.csv")
    print(f"   📄 Individual model predictions: individual_model_predictions.csv")

if __name__ == "__main__":
    main() 