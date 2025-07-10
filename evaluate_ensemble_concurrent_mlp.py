#!/usr/bin/env python3
"""
Ensemble Concurrent Joint MLP Performance Evaluation Script
This script evaluates individual models and ensemble performance during training.

The script generates comprehensive evaluation results including:
- Overall results report (overall_results_report.txt and overall_results.json)
- Detailed prediction CSVs with pixel and mm measurements
- Angle measurements and patient classification analysis
- Visualizations for best/worst performing patients
- Accuracy metrics based on clinical thresholds (2mm/4mm for landmarks, 2°/4° for angles)
- Performance analysis for ALL landmarks and angles (not just key ones)

Key Features:
- Coordinate scaling from 224x224 to 600x600 original image space
- Millimeter calibration using patient-specific ruler measurements
- Clinical accuracy assessment with 2mm/4mm and 2°/4° thresholds
- Comprehensive landmark performance (all 19 landmarks)
- Complete angle analysis (SNA, SNB, ANB, u1, l1, sn_ans_pns, sn_mn_go, nasolabial)
- Patient classification based on ANB angle
- Ensemble vs individual model comparison

Usage examples:
  # Evaluate using best/latest checkpoints (default)
  python evaluate_ensemble_concurrent_mlp.py
  
  # Evaluate all models at epoch 20
  python evaluate_ensemble_concurrent_mlp.py --epoch 20
  
  # Evaluate 5 models at epoch 30 with individual model analysis
  python evaluate_ensemble_concurrent_mlp.py --n_models 5 --epoch 30 --evaluate_individual
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
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
import joblib
from typing import List, Dict, Tuple, Optional
import json

# Add current directory to path for custom modules
sys.path.insert(0, os.getcwd())

# Calibration constants
RULER_LENGTH_MM = 10.0  # Standard cephalometric ruler length in mm

def load_ruler_calibration_data(ruler_data_path: str) -> Dict[str, Dict]:
    """Load ruler calibration data from JSON file."""
    try:
        with open(ruler_data_path, 'r') as f:
            ruler_data = json.load(f)
        print(f"✓ Loaded ruler calibration data for {len(ruler_data)} patients")
        return ruler_data
    except Exception as e:
        print(f"⚠️  Failed to load ruler calibration data: {e}")
        return {}

def calculate_pixel_to_mm_ratio(ruler_data: Dict[str, Dict], patient_id: int) -> Optional[float]:
    """Calculate the pixel to mm conversion ratio for a specific patient."""
    patient_key = str(patient_id)
    
    if patient_key not in ruler_data:
        return None
    
    patient_ruler = ruler_data[patient_key]
    
    # Check if ruler points are valid
    if (patient_ruler.get('ruler_point_up_x') is None or 
        patient_ruler.get('ruler_point_up_y') is None or
        patient_ruler.get('ruler_point_down_x') is None or 
        patient_ruler.get('ruler_point_down_y') is None):
        return None
    
    # Calculate pixel distance between ruler points (in 600x600 space)
    ruler_pixel_distance = np.sqrt(
        (patient_ruler['ruler_point_down_x'] - patient_ruler['ruler_point_up_x'])**2 +
        (patient_ruler['ruler_point_down_y'] - patient_ruler['ruler_point_up_y'])**2
    )
    
    if ruler_pixel_distance == 0:
        return None
    
    # Calculate mm per pixel ratio
    mm_per_pixel_600 = RULER_LENGTH_MM / ruler_pixel_distance
    
    return mm_per_pixel_600

def convert_errors_to_mm(errors_pixels: np.ndarray, mm_per_pixel: float) -> np.ndarray:
    """Convert pixel errors to millimeter errors."""
    return errors_pixels * mm_per_pixel

# Angle calculation functions
def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculate angle at p2 between vectors p2->p1 and p2->p3 in degrees."""
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Calculate angle using dot product
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def ang(line1: List[np.ndarray], line2: List[np.ndarray]) -> float:
    """Calculate angle between two lines defined by their endpoints."""
    # Vector for line 1
    v1 = line1[1] - line1[0]
    # Vector for line 2
    v2 = line2[1] - line2[0]
    
    # Calculate angle using dot product
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def calculate_cephalometric_angles(coords: np.ndarray, landmark_names: List[str]) -> Dict[str, float]:
    """Calculate cephalometric angles from landmark coordinates."""
    # Create landmark index map
    landmark_idx = {name: i for i, name in enumerate(landmark_names)}
    
    # Helper function to get landmark coordinates
    def get_point(name: str) -> np.ndarray:
        if name in landmark_idx:
            return coords[landmark_idx[name]]
        return np.array([0, 0])
    
    angles = {}
    
    # SNA angle
    try:
        s = get_point('sella')
        n = get_point('nasion')
        a = get_point('A_point')
        if np.all(s > 0) and np.all(n > 0) and np.all(a > 0):
            angles['SNA'] = 180 - ang([s, n], [n, a])
        else:
            angles['SNA'] = np.nan
    except:
        angles['SNA'] = np.nan
    
    # SNB angle
    try:
        s = get_point('sella')
        n = get_point('nasion')
        b = get_point('B_point')
        if np.all(s > 0) and np.all(n > 0) and np.all(b > 0):
            angles['SNB'] = 180 - ang([s, n], [n, b])
        else:
            angles['SNB'] = np.nan
    except:
        angles['SNB'] = np.nan
    
    # ANB angle
    try:
        a = get_point('A_point')
        n = get_point('nasion')
        b = get_point('B_point')
        if np.all(a > 0) and np.all(n > 0) and np.all(b > 0):
            if not np.isnan(angles.get('SNA', np.nan)) and not np.isnan(angles.get('SNB', np.nan)):
                if angles['SNA'] - angles['SNB'] > 0:
                    angles['ANB'] = 180 - ang([n, b], [a, n])
                else:
                    angles['ANB'] = -1 * (180 - ang([n, b], [a, n]))
            else:
                angles['ANB'] = np.nan
        else:
            angles['ANB'] = np.nan
    except:
        angles['ANB'] = np.nan
    
    # u1 angle (upper incisor to palatal plane)
    try:
        ans = get_point('ANS')
        pns = get_point('PNS')
        u_tip = get_point('upper_1_tip')
        u_apex = get_point('upper_1_apex')
        if np.all(ans > 0) and np.all(pns > 0) and np.all(u_tip > 0) and np.all(u_apex > 0):
            angles['u1'] = 180 - ang([ans, pns], [u_tip, u_apex])
        else:
            angles['u1'] = np.nan
    except:
        angles['u1'] = np.nan
    
    # l1 angle (lower incisor to mandibular plane)
    try:
        go = get_point('Gonion')
        mn = get_point('Menton')
        l_tip = get_point('lower_1_tip')
        l_apex = get_point('lower_1_apex')
        if np.all(go > 0) and np.all(mn > 0) and np.all(l_tip > 0) and np.all(l_apex > 0):
            angles['l1'] = 180 - ang([go, mn], [l_apex, l_tip])
        else:
            angles['l1'] = np.nan
    except:
        angles['l1'] = np.nan
    
    # SN-ANS/PNS angle (palatal plane to cranial base)
    try:
        ans = get_point('ANS')
        pns = get_point('PNS')
        s = get_point('sella')
        n = get_point('nasion')
        if np.all(ans > 0) and np.all(pns > 0) and np.all(s > 0) and np.all(n > 0):
            angles['sn_ans_pns'] = 180 - ang([s, n], [ans, pns])
        else:
            angles['sn_ans_pns'] = np.nan
    except:
        angles['sn_ans_pns'] = np.nan
    
    # SN-Go/Mn angle (mandibular plane to cranial base)
    try:
        go = get_point('Gonion')
        mn = get_point('Menton')
        s = get_point('sella')
        n = get_point('nasion')
        if np.all(go > 0) and np.all(mn > 0) and np.all(s > 0) and np.all(n > 0):
            angles['sn_mn_go'] = ang([s, n], [go, mn])
        else:
            angles['sn_mn_go'] = np.nan
    except:
        angles['sn_mn_go'] = np.nan
    
    return angles

def calculate_perpendicular_distance(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
    """Calculate perpendicular distance from a point to a line defined by two points."""
    # Vector from line_start to line_end
    line_vec = line_end - line_start
    # Vector from line_start to point
    point_vec = point - line_start
    
    # Project point_vec onto line_vec
    line_length = np.linalg.norm(line_vec)
    if line_length == 0:
        return np.linalg.norm(point_vec)
    
    line_unitvec = line_vec / line_length
    proj_length = np.dot(point_vec, line_unitvec)
    
    # Find the closest point on the line
    if proj_length < 0:
        closest_point = line_start
    elif proj_length > line_length:
        closest_point = line_end
    else:
        closest_point = line_start + proj_length * line_unitvec
    
    # Calculate distance
    return np.linalg.norm(point - closest_point)

def calculate_soft_tissue_measurements(coords: np.ndarray, landmark_names: List[str]) -> Dict[str, float]:
    """Calculate soft tissue measurements including Nasolabial angle and E-line distances."""
    # Create landmark index map
    landmark_idx = {name: i for i, name in enumerate(landmark_names)}
    
    # Helper function to get landmark coordinates
    def get_point(name: str) -> np.ndarray:
        if name in landmark_idx:
            return coords[landmark_idx[name]]
        return np.array([0, 0])
    
    measurements = {}
    
    # Nasolabial Angle
    try:
        tip_nose = get_point('Tip_of_the_nose')
        subnasal = get_point('Subnasal')
        upper_lip = get_point('Upper_lip')
        
        if np.all(tip_nose > 0) and np.all(subnasal > 0) and np.all(upper_lip > 0):
            # Angle at subnasal between tip_nose and upper_lip
            nasolabial = ang([subnasal, tip_nose], [subnasal, upper_lip])
            measurements['nasolabial_angle'] = nasolabial
        else:
            measurements['nasolabial_angle'] = np.nan
    except:
        measurements['nasolabial_angle'] = np.nan
    
    # E-Line measurements
    try:
        tip_nose = get_point('Tip_of_the_nose')
        st_pogonion = get_point('ST_Pogonion')
        upper_lip = get_point('Upper_lip')
        lower_lip = get_point('Lower_lip')
        
        if (np.all(tip_nose > 0) and np.all(st_pogonion > 0) and 
            np.all(upper_lip > 0) and np.all(lower_lip > 0)):
            
            # Calculate perpendicular distances to E-line
            upper_lip_dist = calculate_perpendicular_distance(upper_lip, tip_nose, st_pogonion)
            lower_lip_dist = calculate_perpendicular_distance(lower_lip, tip_nose, st_pogonion)
            
            # Store distances in 224x224 space
            measurements['upper_lip_to_eline'] = upper_lip_dist
            measurements['lower_lip_to_eline'] = lower_lip_dist
        else:
            measurements['upper_lip_to_eline'] = np.nan
            measurements['lower_lip_to_eline'] = np.nan
    except:
        measurements['upper_lip_to_eline'] = np.nan
        measurements['lower_lip_to_eline'] = np.nan
    
    return measurements

def classify_patient(anb_angle: float) -> str:
    """Classify patient based on ANB angle."""
    if np.isnan(anb_angle):
        return 'Unknown'
    
    if 0 < anb_angle < 4:
        return 'Class I'
    elif anb_angle >= 4:
        return 'Class II'
    else:  # anb_angle <= 0
        return 'Class III'

def calculate_classification_metrics(true_labels: List[str], pred_labels: List[str]) -> Dict[str, any]:
    """Calculate classification metrics including accuracy, precision, recall, F1, and confusion matrix."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    # Filter out 'Unknown' classifications
    valid_indices = [(t != 'Unknown' and p != 'Unknown') for t, p in zip(true_labels, pred_labels)]
    true_valid = [t for t, v in zip(true_labels, valid_indices) if v]
    pred_valid = [p for p, v in zip(pred_labels, valid_indices) if v]
    
    if not true_valid:
        return {
            'accuracy': np.nan,
            'precision': np.nan,
            'recall': np.nan,
            'f1_score': np.nan,
            'confusion_matrix': None,
            'n_samples': 0
        }
    
    # Calculate metrics
    classes = ['Class I', 'Class II', 'Class III']
    
    # Ensure all classes are represented
    true_valid_encoded = [classes.index(c) if c in classes else -1 for c in true_valid]
    pred_valid_encoded = [classes.index(c) if c in classes else -1 for c in pred_valid]
    
    accuracy = accuracy_score(true_valid_encoded, pred_valid_encoded)
    
    # Calculate per-class metrics with macro averaging
    precision = precision_score(true_valid_encoded, pred_valid_encoded, average='macro', zero_division=0)
    recall = recall_score(true_valid_encoded, pred_valid_encoded, average='macro', zero_division=0)
    f1 = f1_score(true_valid_encoded, pred_valid_encoded, average='macro', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(true_valid_encoded, pred_valid_encoded, labels=[0, 1, 2])
    
    # Also calculate per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(classes):
        class_true = [1 if t == i else 0 for t in true_valid_encoded]
        class_pred = [1 if p == i else 0 for p in pred_valid_encoded]
        
        if sum(class_true) > 0:  # If this class exists in ground truth
            per_class_metrics[class_name] = {
                'precision': precision_score(class_true, class_pred, zero_division=0),
                'recall': recall_score(class_true, class_pred, zero_division=0),
                'f1': f1_score(class_true, class_pred, zero_division=0),
                'support': sum(class_true)
            }
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'class_names': classes,
        'per_class': per_class_metrics,
        'n_samples': len(true_valid)
    }

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

# Image scaling constants
ORIGINAL_IMAGE_SIZE = 600
MODEL_INPUT_SIZE = 224
SCALE_FACTOR = ORIGINAL_IMAGE_SIZE / MODEL_INPUT_SIZE  # 2.6786

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

def load_model_components(model_dir: str, device: torch.device, config_path: str, epoch: Optional[int] = None) -> Optional[Tuple]:
    """Load HRNet model, MLP model, and scalers for a single ensemble model."""
    print(f"\n🔄 Loading model from: {os.path.basename(model_dir)}")
    
    # Find HRNet checkpoint
    if epoch is not None:
        # Look for specific epoch checkpoint
        hrnet_checkpoint = os.path.join(model_dir, f"epoch_{epoch}.pth")
        if not os.path.exists(hrnet_checkpoint):
            # Show available epochs for better debugging
            available_epochs = []
            epoch_pattern = os.path.join(model_dir, "epoch_*.pth")
            epoch_files = glob.glob(epoch_pattern)
            for ep_file in epoch_files:
                try:
                    ep_num = int(os.path.basename(ep_file).split("epoch_")[1].split(".")[0])
                    available_epochs.append(ep_num)
                except:
                    pass
            available_epochs.sort()
            
            print(f"   ❌ Epoch {epoch} checkpoint not found: {hrnet_checkpoint}")
            if available_epochs:
                print(f"      Available epochs: {available_epochs}")
            else:
                print(f"      No epoch checkpoints found in {model_dir}")
            return None
        hrnet_checkpoint_name = os.path.basename(hrnet_checkpoint)
        print(f"   ✓ HRNet checkpoint (epoch {epoch}): {hrnet_checkpoint_name}")
    else:
        # Use existing logic for best/latest checkpoint
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
    
    if epoch is not None:
        # Look for specific epoch MLP checkpoint
        epoch_mlp_path = os.path.join(mlp_dir, f"mlp_joint_epoch_{epoch}.pth")
        if os.path.exists(epoch_mlp_path):
            synchronized_mlp_path = epoch_mlp_path
            model_type = f"epoch_{epoch}"
            print(f"   ✓ MLP checkpoint (epoch {epoch}): {os.path.basename(epoch_mlp_path)}")
        else:
            # Show available MLP epochs for better debugging
            available_mlp_epochs = []
            mlp_epoch_pattern = os.path.join(mlp_dir, "mlp_joint_epoch_*.pth")
            mlp_epoch_files = glob.glob(mlp_epoch_pattern)
            for mlp_file in mlp_epoch_files:
                try:
                    ep_num = int(os.path.basename(mlp_file).split("_epoch_")[1].split(".")[0])
                    available_mlp_epochs.append(ep_num)
                except:
                    pass
            available_mlp_epochs.sort()
            
            print(f"   ❌ Epoch {epoch} MLP checkpoint not found: {epoch_mlp_path}")
            if available_mlp_epochs:
                print(f"      Available MLP epochs: {available_mlp_epochs}")
            else:
                print(f"      No MLP epoch checkpoints found in {mlp_dir}")
            return None
    else:
        # Use existing logic for synchronized/fallback checkpoints
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
                         test_df, landmark_names, landmark_cols, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Evaluate a single model and return predictions with patient IDs."""
    hrnet_predictions = []
    mlp_predictions = []
    ground_truths = []
    patient_ids = []
    
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
            patient_ids.append(row['patient_id'])
            
        except Exception as e:
            continue
    
    if len(hrnet_predictions) == 0:
        return None, None, None, []
    
    return np.array(hrnet_predictions), np.array(mlp_predictions), np.array(ground_truths), patient_ids

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

def compute_metrics_with_mm(pred_coords, gt_coords, landmark_names, patient_ids, ruler_data) -> Tuple[Dict, Dict]:
    """Compute evaluation metrics including mm conversions when calibration is available."""
    # First compute pixel-based metrics
    overall_metrics, per_landmark_metrics = compute_metrics(pred_coords, gt_coords, landmark_names)
    
    # Add mm conversions if ruler data is available
    if ruler_data and len(patient_ids) > 0:
        # Compute per-patient mm errors
        all_mm_errors = []
        calibrated_patients = 0
        
        # Initialize accuracy counters for overall metrics
        overall_2mm_accurate = 0
        overall_4mm_accurate = 0
        overall_total_predictions = 0
        
        for i, patient_id in enumerate(patient_ids):
            mm_per_pixel_600 = calculate_pixel_to_mm_ratio(ruler_data, patient_id)
            if mm_per_pixel_600:
                mm_per_pixel_224 = mm_per_pixel_600 * SCALE_FACTOR
                calibrated_patients += 1
                
                # Convert patient errors to mm
                valid_mask = (gt_coords[i, :, 0] > 0) & (gt_coords[i, :, 1] > 0)
                if np.any(valid_mask):
                    pixel_errors = np.sqrt(np.sum((pred_coords[i, valid_mask] - gt_coords[i, valid_mask])**2, axis=1))
                    mm_errors = pixel_errors * mm_per_pixel_224
                    all_mm_errors.extend(mm_errors)
                    
                    # Count overall accuracy
                    overall_2mm_accurate += np.sum(mm_errors <= 2.0)
                    overall_4mm_accurate += np.sum(mm_errors <= 4.0)
                    overall_total_predictions += len(mm_errors)
        
        # Add mm statistics to overall metrics
        if all_mm_errors:
            overall_metrics['mre_mm'] = np.mean(all_mm_errors)
            overall_metrics['std_mm'] = np.std(all_mm_errors)
            overall_metrics['median_mm'] = np.median(all_mm_errors)
            overall_metrics['p90_mm'] = np.percentile(all_mm_errors, 90)
            overall_metrics['p95_mm'] = np.percentile(all_mm_errors, 95)
            overall_metrics['max_mm'] = np.max(all_mm_errors)
            overall_metrics['calibrated_patients'] = calibrated_patients
            
            # Add overall accuracy metrics
            overall_metrics['accuracy_2mm'] = overall_2mm_accurate / overall_total_predictions if overall_total_predictions > 0 else 0.0
            overall_metrics['accuracy_4mm'] = overall_4mm_accurate / overall_total_predictions if overall_total_predictions > 0 else 0.0
            overall_metrics['accuracy_2mm_count'] = overall_2mm_accurate
            overall_metrics['accuracy_4mm_count'] = overall_4mm_accurate
            overall_metrics['accuracy_total_count'] = overall_total_predictions
            
            # Add per-landmark mm metrics and accuracy
            for j, name in enumerate(landmark_names):
                landmark_mm_errors = []
                landmark_2mm_accurate = 0
                landmark_4mm_accurate = 0
                landmark_total = 0
                
                for i, patient_id in enumerate(patient_ids):
                    mm_per_pixel_600 = calculate_pixel_to_mm_ratio(ruler_data, patient_id)
                    if mm_per_pixel_600:
                        mm_per_pixel_224 = mm_per_pixel_600 * SCALE_FACTOR
                        if gt_coords[i, j, 0] > 0 and gt_coords[i, j, 1] > 0:
                            pixel_error = np.sqrt(np.sum((pred_coords[i, j] - gt_coords[i, j])**2))
                            mm_error = pixel_error * mm_per_pixel_224
                            landmark_mm_errors.append(mm_error)
                            
                            # Count accuracy for this landmark
                            if mm_error <= 2.0:
                                landmark_2mm_accurate += 1
                            if mm_error <= 4.0:
                                landmark_4mm_accurate += 1
                            landmark_total += 1
                
                if landmark_mm_errors:
                    per_landmark_metrics[name]['mre_mm'] = np.mean(landmark_mm_errors)
                    per_landmark_metrics[name]['std_mm'] = np.std(landmark_mm_errors)
                    per_landmark_metrics[name]['median_mm'] = np.median(landmark_mm_errors)
                    
                    # Add per-landmark accuracy metrics
                    per_landmark_metrics[name]['accuracy_2mm'] = float(landmark_2mm_accurate / landmark_total) if landmark_total > 0 else 0.0
                    per_landmark_metrics[name]['accuracy_4mm'] = float(landmark_4mm_accurate / landmark_total) if landmark_total > 0 else 0.0
                    per_landmark_metrics[name]['accuracy_2mm_count'] = int(landmark_2mm_accurate)
                    per_landmark_metrics[name]['accuracy_4mm_count'] = int(landmark_4mm_accurate)
                    per_landmark_metrics[name]['accuracy_total_count'] = int(landmark_total)
    
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

def save_ensemble_predictions_to_csv(ensemble_hrnet: np.ndarray, ensemble_mlp: np.ndarray,
                                   gt_coords: np.ndarray, patient_ids: List[int],
                                   landmark_names: List[str], output_dir: str,
                                   ruler_data: Optional[Dict[str, Dict]] = None):
    """Save ensemble predictions to CSV files with detailed per-landmark information."""
    print(f"\n💾 Saving detailed predictions to CSV...")
    
    # Prepare data for both CSV files
    mlp_data = []
    hrnet_data = []
    
    # Statistics for mm conversions
    mm_conversion_count = 0
    missing_calibration_count = 0
    
    for i, patient_id in enumerate(patient_ids):
        mlp_row = {'patient_id': patient_id}
        hrnet_row = {'patient_id': patient_id}
        
        # Calculate pixel to mm ratio for this patient
        mm_per_pixel_600 = None
        mm_per_pixel_224 = None
        if ruler_data:
            mm_per_pixel_600 = calculate_pixel_to_mm_ratio(ruler_data, patient_id)
            if mm_per_pixel_600:
                mm_per_pixel_224 = mm_per_pixel_600 * SCALE_FACTOR
                mm_conversion_count += 1
            else:
                missing_calibration_count += 1
        
        # Add ground truth and predictions for each landmark
        for j, landmark in enumerate(landmark_names):
            # Ground truth (scale to original 600x600)
            gt_x = gt_coords[i, j, 0]
            gt_y = gt_coords[i, j, 1]
            gt_x_scaled = gt_x * SCALE_FACTOR if gt_x > 0 else 0
            gt_y_scaled = gt_y * SCALE_FACTOR if gt_y > 0 else 0
            
            mlp_row[f'gt_{landmark}_x'] = gt_x_scaled
            mlp_row[f'gt_{landmark}_y'] = gt_y_scaled
            hrnet_row[f'gt_{landmark}_x'] = gt_x_scaled
            hrnet_row[f'gt_{landmark}_y'] = gt_y_scaled
            
            # Ensemble MLP predictions (scale to original 600x600)
            mlp_x = ensemble_mlp[i, j, 0]
            mlp_y = ensemble_mlp[i, j, 1]
            mlp_x_scaled = mlp_x * SCALE_FACTOR
            mlp_y_scaled = mlp_y * SCALE_FACTOR
            
            # Error in both 224x224 and 600x600 scales
            mlp_error_224 = np.sqrt((mlp_x - gt_x)**2 + (mlp_y - gt_y)**2) if gt_x > 0 and gt_y > 0 else np.nan
            mlp_error_600 = mlp_error_224 * SCALE_FACTOR if not np.isnan(mlp_error_224) else np.nan
            
            mlp_row[f'ensemble_mlp_{landmark}_x'] = mlp_x_scaled
            mlp_row[f'ensemble_mlp_{landmark}_y'] = mlp_y_scaled
            mlp_row[f'ensemble_mlp_{landmark}_error_224px'] = mlp_error_224
            mlp_row[f'ensemble_mlp_{landmark}_error_600px'] = mlp_error_600
            
            # Add mm errors if calibration is available
            if mm_per_pixel_224 and mm_per_pixel_600:
                mlp_error_224_mm = mlp_error_224 * mm_per_pixel_224 if not np.isnan(mlp_error_224) else np.nan
                mlp_error_600_mm = mlp_error_600 * mm_per_pixel_600 if not np.isnan(mlp_error_600) else np.nan
                mlp_row[f'ensemble_mlp_{landmark}_error_mm'] = mlp_error_600_mm
            
            # Ensemble HRNet predictions (scale to original 600x600)
            hrnet_x = ensemble_hrnet[i, j, 0]
            hrnet_y = ensemble_hrnet[i, j, 1]
            hrnet_x_scaled = hrnet_x * SCALE_FACTOR
            hrnet_y_scaled = hrnet_y * SCALE_FACTOR
            
            # Error in both 224x224 and 600x600 scales
            hrnet_error_224 = np.sqrt((hrnet_x - gt_x)**2 + (hrnet_y - gt_y)**2) if gt_x > 0 and gt_y > 0 else np.nan
            hrnet_error_600 = hrnet_error_224 * SCALE_FACTOR if not np.isnan(hrnet_error_224) else np.nan
            
            hrnet_row[f'ensemble_hrnetv2_{landmark}_x'] = hrnet_x_scaled
            hrnet_row[f'ensemble_hrnetv2_{landmark}_y'] = hrnet_y_scaled
            hrnet_row[f'ensemble_hrnetv2_{landmark}_error_224px'] = hrnet_error_224
            hrnet_row[f'ensemble_hrnetv2_{landmark}_error_600px'] = hrnet_error_600
            
            # Add mm errors if calibration is available
            if mm_per_pixel_224 and mm_per_pixel_600:
                hrnet_error_224_mm = hrnet_error_224 * mm_per_pixel_224 if not np.isnan(hrnet_error_224) else np.nan
                hrnet_error_600_mm = hrnet_error_600 * mm_per_pixel_600 if not np.isnan(hrnet_error_600) else np.nan
                hrnet_row[f'ensemble_hrnetv2_{landmark}_error_mm'] = hrnet_error_600_mm
        
        mlp_data.append(mlp_row)
        hrnet_data.append(hrnet_row)
    
    # Create DataFrames and save to CSV
    mlp_df = pd.DataFrame(mlp_data)
    hrnet_df = pd.DataFrame(hrnet_data)
    
    # Save files
    mlp_csv_path = os.path.join(output_dir, "ensemble_mlp_predictions_detailed.csv")
    hrnet_csv_path = os.path.join(output_dir, "ensemble_hrnetv2_predictions_detailed.csv")
    
    mlp_df.to_csv(mlp_csv_path, index=False)
    hrnet_df.to_csv(hrnet_csv_path, index=False)
    
    print(f"   ✓ Ensemble MLP predictions saved to: {mlp_csv_path}")
    print(f"   ✓ Ensemble HRNetV2 predictions saved to: {hrnet_csv_path}")
    
    # Print summary statistics
    print(f"\n📊 Summary:")
    print(f"   - Total patients: {len(patient_ids)}")
    print(f"   - Total landmarks per patient: {len(landmark_names)}")
    if ruler_data:
        print(f"   - Patients with mm calibration: {mm_conversion_count}/{len(patient_ids)}")
        if missing_calibration_count > 0:
            print(f"   - Patients missing calibration: {missing_calibration_count}")
    
    # Calculate overall mean errors
    mlp_errors_600 = []
    hrnet_errors_600 = []
    mlp_errors_mm = []
    hrnet_errors_mm = []
    
    for col in mlp_df.columns:
        if col.endswith('_error_600px'):
            mlp_errors_600.extend(mlp_df[col].dropna().values)
        elif col.endswith('_error_mm'):
            mlp_errors_mm.extend(mlp_df[col].dropna().values)
    
    for col in hrnet_df.columns:
        if col.endswith('_error_600px'):
            hrnet_errors_600.extend(hrnet_df[col].dropna().values)
        elif col.endswith('_error_mm'):
            hrnet_errors_mm.extend(hrnet_df[col].dropna().values)
    
    if mlp_errors_600:
        print(f"   - Ensemble MLP mean error: {np.mean(mlp_errors_600):.3f} pixels (600x600)")
    if hrnet_errors_600:
        print(f"   - Ensemble HRNetV2 mean error: {np.mean(hrnet_errors_600):.3f} pixels (600x600)")
    
    if mlp_errors_mm:
        print(f"   - Ensemble MLP mean error: {np.mean(mlp_errors_mm):.3f} mm")
    if hrnet_errors_mm:
        print(f"   - Ensemble HRNetV2 mean error: {np.mean(hrnet_errors_mm):.3f} mm")

def save_individual_model_predictions(model_idx: int, hrnet_preds: np.ndarray, mlp_preds: np.ndarray,
                                    gt_coords: np.ndarray, patient_ids: List[int],
                                    landmark_names: List[str], output_dir: str,
                                    ruler_data: Optional[Dict[str, Dict]] = None):
    """Save individual model predictions to CSV files."""
    print(f"   💾 Saving Model {model_idx} predictions...")
    
    # Prepare data for both CSV files
    mlp_data = []
    hrnet_data = []
    
    for i, patient_id in enumerate(patient_ids):
        mlp_row = {'patient_id': patient_id}
        hrnet_row = {'patient_id': patient_id}
        
        # Calculate pixel to mm ratio for this patient
        mm_per_pixel_600 = None
        mm_per_pixel_224 = None
        if ruler_data:
            mm_per_pixel_600 = calculate_pixel_to_mm_ratio(ruler_data, patient_id)
            if mm_per_pixel_600:
                mm_per_pixel_224 = mm_per_pixel_600 * SCALE_FACTOR
        
        # Add ground truth and predictions for each landmark
        for j, landmark in enumerate(landmark_names):
            # Ground truth (scale to original 600x600)
            gt_x = gt_coords[i, j, 0]
            gt_y = gt_coords[i, j, 1]
            gt_x_scaled = gt_x * SCALE_FACTOR if gt_x > 0 else 0
            gt_y_scaled = gt_y * SCALE_FACTOR if gt_y > 0 else 0
            
            mlp_row[f'gt_{landmark}_x'] = gt_x_scaled
            mlp_row[f'gt_{landmark}_y'] = gt_y_scaled
            hrnet_row[f'gt_{landmark}_x'] = gt_x_scaled
            hrnet_row[f'gt_{landmark}_y'] = gt_y_scaled
            
            # Model MLP predictions (scale to original 600x600)
            mlp_x = mlp_preds[i, j, 0]
            mlp_y = mlp_preds[i, j, 1]
            mlp_x_scaled = mlp_x * SCALE_FACTOR
            mlp_y_scaled = mlp_y * SCALE_FACTOR
            
            # Error in both scales
            mlp_error_224 = np.sqrt((mlp_x - gt_x)**2 + (mlp_y - gt_y)**2) if gt_x > 0 and gt_y > 0 else np.nan
            mlp_error_600 = mlp_error_224 * SCALE_FACTOR if not np.isnan(mlp_error_224) else np.nan
            
            mlp_row[f'model{model_idx}_mlp_{landmark}_x'] = mlp_x_scaled
            mlp_row[f'model{model_idx}_mlp_{landmark}_y'] = mlp_y_scaled
            mlp_row[f'model{model_idx}_mlp_{landmark}_error_224px'] = mlp_error_224
            mlp_row[f'model{model_idx}_mlp_{landmark}_error_600px'] = mlp_error_600
            
            # Add mm errors if calibration is available
            if mm_per_pixel_224 and mm_per_pixel_600:
                mlp_error_mm = mlp_error_600 * mm_per_pixel_600 if not np.isnan(mlp_error_600) else np.nan
                mlp_row[f'model{model_idx}_mlp_{landmark}_error_mm'] = mlp_error_mm
            
            # Model HRNet predictions (scale to original 600x600)
            hrnet_x = hrnet_preds[i, j, 0]
            hrnet_y = hrnet_preds[i, j, 1]
            hrnet_x_scaled = hrnet_x * SCALE_FACTOR
            hrnet_y_scaled = hrnet_y * SCALE_FACTOR
            
            # Error in both scales
            hrnet_error_224 = np.sqrt((hrnet_x - gt_x)**2 + (hrnet_y - gt_y)**2) if gt_x > 0 and gt_y > 0 else np.nan
            hrnet_error_600 = hrnet_error_224 * SCALE_FACTOR if not np.isnan(hrnet_error_224) else np.nan
            
            hrnet_row[f'model{model_idx}_hrnetv2_{landmark}_x'] = hrnet_x_scaled
            hrnet_row[f'model{model_idx}_hrnetv2_{landmark}_y'] = hrnet_y_scaled
            hrnet_row[f'model{model_idx}_hrnetv2_{landmark}_error_224px'] = hrnet_error_224
            hrnet_row[f'model{model_idx}_hrnetv2_{landmark}_error_600px'] = hrnet_error_600
            
            # Add mm errors if calibration is available
            if mm_per_pixel_224 and mm_per_pixel_600:
                hrnet_error_mm = hrnet_error_600 * mm_per_pixel_600 if not np.isnan(hrnet_error_600) else np.nan
                hrnet_row[f'model{model_idx}_hrnetv2_{landmark}_error_mm'] = hrnet_error_mm
        
        mlp_data.append(mlp_row)
        hrnet_data.append(hrnet_row)
    
    # Create DataFrames and save to CSV
    mlp_df = pd.DataFrame(mlp_data)
    hrnet_df = pd.DataFrame(hrnet_data)
    
    # Save files
    mlp_csv_path = os.path.join(output_dir, f"model{model_idx}_mlp_predictions_detailed.csv")
    hrnet_csv_path = os.path.join(output_dir, f"model{model_idx}_hrnetv2_predictions_detailed.csv")
    
    mlp_df.to_csv(mlp_csv_path, index=False)
    hrnet_df.to_csv(hrnet_csv_path, index=False)
    
    print(f"      ✓ Model {model_idx} MLP predictions saved")
    print(f"      ✓ Model {model_idx} HRNet predictions saved")

def save_all_models_combined(all_hrnet_preds: List[np.ndarray], all_mlp_preds: List[np.ndarray],
                           ensemble_hrnet: np.ndarray, ensemble_mlp: np.ndarray,
                           gt_coords: np.ndarray, patient_ids: List[int],
                           landmark_names: List[str], output_dir: str,
                           ruler_data: Optional[Dict[str, Dict]] = None):
    """Save all models and ensemble predictions in combined CSV files."""
    print(f"\n💾 Creating combined prediction files...")
    
    # Prepare data for combined CSV files
    combined_mlp_data = []
    combined_hrnet_data = []
    
    for i, patient_id in enumerate(patient_ids):
        mlp_row = {'patient_id': patient_id}
        hrnet_row = {'patient_id': patient_id}
        
        # Calculate pixel to mm ratio for this patient
        mm_per_pixel_600 = None
        mm_per_pixel_224 = None
        if ruler_data:
            mm_per_pixel_600 = calculate_pixel_to_mm_ratio(ruler_data, patient_id)
            if mm_per_pixel_600:
                mm_per_pixel_224 = mm_per_pixel_600 * SCALE_FACTOR
        
        # For each landmark
        for j, landmark in enumerate(landmark_names):
            # Ground truth (scale to original 600x600)
            gt_x = gt_coords[i, j, 0]
            gt_y = gt_coords[i, j, 1]
            gt_x_scaled = gt_x * SCALE_FACTOR if gt_x > 0 else 0
            gt_y_scaled = gt_y * SCALE_FACTOR if gt_y > 0 else 0
            
            mlp_row[f'gt_{landmark}_x'] = gt_x_scaled
            mlp_row[f'gt_{landmark}_y'] = gt_y_scaled
            hrnet_row[f'gt_{landmark}_x'] = gt_x_scaled
            hrnet_row[f'gt_{landmark}_y'] = gt_y_scaled
            
            # Individual model predictions
            for model_idx in range(len(all_hrnet_preds)):
                # MLP predictions (scale to original 600x600)
                mlp_x = all_mlp_preds[model_idx][i, j, 0]
                mlp_y = all_mlp_preds[model_idx][i, j, 1]
                mlp_x_scaled = mlp_x * SCALE_FACTOR
                mlp_y_scaled = mlp_y * SCALE_FACTOR
                
                mlp_error_224 = np.sqrt((mlp_x - gt_x)**2 + (mlp_y - gt_y)**2) if gt_x > 0 and gt_y > 0 else np.nan
                mlp_error_600 = mlp_error_224 * SCALE_FACTOR if not np.isnan(mlp_error_224) else np.nan
                
                mlp_row[f'model{model_idx+1}_mlp_{landmark}_x'] = mlp_x_scaled
                mlp_row[f'model{model_idx+1}_mlp_{landmark}_y'] = mlp_y_scaled
                mlp_row[f'model{model_idx+1}_mlp_{landmark}_error_224px'] = mlp_error_224
                mlp_row[f'model{model_idx+1}_mlp_{landmark}_error_600px'] = mlp_error_600
                
                # Add mm error if calibration is available
                if mm_per_pixel_600:
                    mlp_error_mm = mlp_error_600 * mm_per_pixel_600 if not np.isnan(mlp_error_600) else np.nan
                    mlp_row[f'model{model_idx+1}_mlp_{landmark}_error_mm'] = mlp_error_mm
                
                # HRNet predictions (scale to original 600x600)
                hrnet_x = all_hrnet_preds[model_idx][i, j, 0]
                hrnet_y = all_hrnet_preds[model_idx][i, j, 1]
                hrnet_x_scaled = hrnet_x * SCALE_FACTOR
                hrnet_y_scaled = hrnet_y * SCALE_FACTOR
                
                hrnet_error_224 = np.sqrt((hrnet_x - gt_x)**2 + (hrnet_y - gt_y)**2) if gt_x > 0 and gt_y > 0 else np.nan
                hrnet_error_600 = hrnet_error_224 * SCALE_FACTOR if not np.isnan(hrnet_error_224) else np.nan
                
                hrnet_row[f'model{model_idx+1}_hrnetv2_{landmark}_x'] = hrnet_x_scaled
                hrnet_row[f'model{model_idx+1}_hrnetv2_{landmark}_y'] = hrnet_y_scaled
                hrnet_row[f'model{model_idx+1}_hrnetv2_{landmark}_error_224px'] = hrnet_error_224
                hrnet_row[f'model{model_idx+1}_hrnetv2_{landmark}_error_600px'] = hrnet_error_600
                
                # Add mm error if calibration is available
                if mm_per_pixel_600:
                    hrnet_error_mm = hrnet_error_600 * mm_per_pixel_600 if not np.isnan(hrnet_error_600) else np.nan
                    hrnet_row[f'model{model_idx+1}_hrnetv2_{landmark}_error_mm'] = hrnet_error_mm
            
            # Ensemble predictions (scale to original 600x600)
            # MLP ensemble
            ens_mlp_x = ensemble_mlp[i, j, 0]
            ens_mlp_y = ensemble_mlp[i, j, 1]
            ens_mlp_x_scaled = ens_mlp_x * SCALE_FACTOR
            ens_mlp_y_scaled = ens_mlp_y * SCALE_FACTOR
            
            ens_mlp_error_224 = np.sqrt((ens_mlp_x - gt_x)**2 + (ens_mlp_y - gt_y)**2) if gt_x > 0 and gt_y > 0 else np.nan
            ens_mlp_error_600 = ens_mlp_error_224 * SCALE_FACTOR if not np.isnan(ens_mlp_error_224) else np.nan
            
            mlp_row[f'ensemble_mlp_{landmark}_x'] = ens_mlp_x_scaled
            mlp_row[f'ensemble_mlp_{landmark}_y'] = ens_mlp_y_scaled
            mlp_row[f'ensemble_mlp_{landmark}_error_224px'] = ens_mlp_error_224
            mlp_row[f'ensemble_mlp_{landmark}_error_600px'] = ens_mlp_error_600
            
            # Add mm error if calibration is available
            if mm_per_pixel_600:
                ens_mlp_error_mm = ens_mlp_error_600 * mm_per_pixel_600 if not np.isnan(ens_mlp_error_600) else np.nan
                mlp_row[f'ensemble_mlp_{landmark}_error_mm'] = ens_mlp_error_mm
            
            # HRNet ensemble
            ens_hrnet_x = ensemble_hrnet[i, j, 0]
            ens_hrnet_y = ensemble_hrnet[i, j, 1]
            ens_hrnet_x_scaled = ens_hrnet_x * SCALE_FACTOR
            ens_hrnet_y_scaled = ens_hrnet_y * SCALE_FACTOR
            
            ens_hrnet_error_224 = np.sqrt((ens_hrnet_x - gt_x)**2 + (ens_hrnet_y - gt_y)**2) if gt_x > 0 and gt_y > 0 else np.nan
            ens_hrnet_error_600 = ens_hrnet_error_224 * SCALE_FACTOR if not np.isnan(ens_hrnet_error_224) else np.nan
            
            hrnet_row[f'ensemble_hrnetv2_{landmark}_x'] = ens_hrnet_x_scaled
            hrnet_row[f'ensemble_hrnetv2_{landmark}_y'] = ens_hrnet_y_scaled
            hrnet_row[f'ensemble_hrnetv2_{landmark}_error_224px'] = ens_hrnet_error_224
            hrnet_row[f'ensemble_hrnetv2_{landmark}_error_600px'] = ens_hrnet_error_600
            
            # Add mm error if calibration is available
            if mm_per_pixel_600:
                ens_hrnet_error_mm = ens_hrnet_error_600 * mm_per_pixel_600 if not np.isnan(ens_hrnet_error_600) else np.nan
                hrnet_row[f'ensemble_hrnetv2_{landmark}_error_mm'] = ens_hrnet_error_mm
        
        combined_mlp_data.append(mlp_row)
        combined_hrnet_data.append(hrnet_row)
    
    # Create DataFrames and save to CSV
    combined_mlp_df = pd.DataFrame(combined_mlp_data)
    combined_hrnet_df = pd.DataFrame(combined_hrnet_data)
    
    # Save files
    combined_mlp_csv_path = os.path.join(output_dir, "all_models_mlp_predictions_combined.csv")
    combined_hrnet_csv_path = os.path.join(output_dir, "all_models_hrnetv2_predictions_combined.csv")
    
    combined_mlp_df.to_csv(combined_mlp_csv_path, index=False)
    combined_hrnet_df.to_csv(combined_hrnet_csv_path, index=False)
    
    print(f"   ✓ Combined MLP predictions saved to: {os.path.basename(combined_mlp_csv_path)}")
    print(f"   ✓ Combined HRNetV2 predictions saved to: {os.path.basename(combined_hrnet_csv_path)}")
    
    # Print model diversity statistics
    print(f"\n📊 Model Diversity Analysis:")
    
    # Calculate standard deviation across models for each prediction
    model_stds = []
    for i in range(len(patient_ids)):
        for j in range(len(landmark_names)):
            # Skip invalid landmarks
            if gt_coords[i, j, 0] > 0 and gt_coords[i, j, 1] > 0:
                # MLP predictions across models
                mlp_preds_x = [all_mlp_preds[m][i, j, 0] for m in range(len(all_mlp_preds))]
                mlp_preds_y = [all_mlp_preds[m][i, j, 1] for m in range(len(all_mlp_preds))]
                
                std_x = np.std(mlp_preds_x)
                std_y = np.std(mlp_preds_y)
                model_stds.append(np.sqrt(std_x**2 + std_y**2))
    
    if model_stds:
        print(f"   - Mean prediction std across models: {np.mean(model_stds):.3f} pixels")
        print(f"   - Max prediction std across models: {np.max(model_stds):.3f} pixels")
        print(f"   - Models show {'good' if np.mean(model_stds) > 0.5 else 'limited'} diversity")

def save_angle_predictions_to_csv(ensemble_hrnet: np.ndarray, ensemble_mlp: np.ndarray,
                                 all_hrnet_preds: List[np.ndarray], all_mlp_preds: List[np.ndarray],
                                 gt_coords: np.ndarray, patient_ids: List[int],
                                 landmark_names: List[str], output_dir: str,
                                 ruler_data: Optional[Dict[str, Dict]] = None):
    """Save cephalometric angle calculations, soft tissue measurements, and patient classification to CSV files."""
    print(f"\n📐 Calculating and saving cephalometric angles and measurements...")
    
    # Prepare data for angle CSV files
    ensemble_angle_data = []
    individual_angle_data = []
    angle_names = ['SNA', 'SNB', 'ANB', 'u1', 'l1', 'sn_ans_pns', 'sn_mn_go']
    soft_tissue_names = ['nasolabial_angle', 'upper_lip_to_eline', 'lower_lip_to_eline']
    
    # Lists for classification analysis
    gt_classifications = []
    ensemble_hrnet_classifications = []
    ensemble_mlp_classifications = []
    individual_model_classifications = [[] for _ in range(len(all_hrnet_preds))]
    
    # Process each patient
    for i, patient_id in enumerate(patient_ids):
        # Calculate pixel to mm ratio for this patient
        mm_per_pixel_224 = None
        if ruler_data:
            mm_per_pixel_600 = calculate_pixel_to_mm_ratio(ruler_data, patient_id)
            if mm_per_pixel_600:
                mm_per_pixel_224 = mm_per_pixel_600 * SCALE_FACTOR
        
        # Calculate ground truth angles
        gt_angles = calculate_cephalometric_angles(gt_coords[i], landmark_names)
        gt_soft_tissue = calculate_soft_tissue_measurements(gt_coords[i], landmark_names)
        
        # Calculate ensemble predictions angles
        ensemble_hrnet_angles = calculate_cephalometric_angles(ensemble_hrnet[i], landmark_names)
        ensemble_hrnet_soft_tissue = calculate_soft_tissue_measurements(ensemble_hrnet[i], landmark_names)
        
        ensemble_mlp_angles = calculate_cephalometric_angles(ensemble_mlp[i], landmark_names)
        ensemble_mlp_soft_tissue = calculate_soft_tissue_measurements(ensemble_mlp[i], landmark_names)
        
        # Patient classification based on ANB angle
        gt_anb = gt_angles.get('ANB', np.nan)
        gt_class = classify_patient(gt_anb)
        gt_classifications.append(gt_class)
        
        ensemble_hrnet_anb = ensemble_hrnet_angles.get('ANB', np.nan)
        ensemble_hrnet_class = classify_patient(ensemble_hrnet_anb)
        ensemble_hrnet_classifications.append(ensemble_hrnet_class)
        
        ensemble_mlp_anb = ensemble_mlp_angles.get('ANB', np.nan)
        ensemble_mlp_class = classify_patient(ensemble_mlp_anb)
        ensemble_mlp_classifications.append(ensemble_mlp_class)
        
        # Prepare ensemble angle row
        ensemble_row = {'patient_id': patient_id}
        
        # Add ground truth and predictions for each angle
        for angle_name in angle_names:
            # Ground truth
            gt_angle = gt_angles.get(angle_name, np.nan)
            ensemble_row[f'gt_{angle_name}'] = gt_angle
            
            # Ensemble HRNetV2
            hrnet_angle = ensemble_hrnet_angles.get(angle_name, np.nan)
            hrnet_error = abs(hrnet_angle - gt_angle) if not np.isnan(gt_angle) and not np.isnan(hrnet_angle) else np.nan
            ensemble_row[f'ensemble_hrnetv2_{angle_name}'] = hrnet_angle
            ensemble_row[f'ensemble_hrnetv2_{angle_name}_error'] = hrnet_error
            
            # Ensemble MLP
            mlp_angle = ensemble_mlp_angles.get(angle_name, np.nan)
            mlp_error = abs(mlp_angle - gt_angle) if not np.isnan(gt_angle) and not np.isnan(mlp_angle) else np.nan
            ensemble_row[f'ensemble_mlp_{angle_name}'] = mlp_angle
            ensemble_row[f'ensemble_mlp_{angle_name}_error'] = mlp_error
        
        # Add soft tissue measurements
        for st_name in soft_tissue_names:
            # Ground truth
            gt_st = gt_soft_tissue.get(st_name, np.nan)
            
            # Scale E-line distances to 600x600, but not angles
            if 'eline' in st_name and not np.isnan(gt_st):
                gt_st_scaled = gt_st * SCALE_FACTOR
                ensemble_row[f'gt_{st_name}_224px'] = gt_st
                ensemble_row[f'gt_{st_name}_600px'] = gt_st_scaled
                
                # Add mm measurement if calibration is available
                if mm_per_pixel_224:
                    gt_st_mm = gt_st * mm_per_pixel_224
                    ensemble_row[f'gt_{st_name}_mm'] = gt_st_mm
            else:
                ensemble_row[f'gt_{st_name}'] = gt_st
            
            # Ensemble HRNetV2
            hrnet_st = ensemble_hrnet_soft_tissue.get(st_name, np.nan)
            
            if 'eline' in st_name and not np.isnan(hrnet_st):
                hrnet_st_scaled = hrnet_st * SCALE_FACTOR
                hrnet_st_error_224 = abs(hrnet_st - gt_st) if not np.isnan(gt_st) else np.nan
                hrnet_st_error_600 = hrnet_st_error_224 * SCALE_FACTOR if not np.isnan(hrnet_st_error_224) else np.nan
                
                ensemble_row[f'ensemble_hrnetv2_{st_name}_224px'] = hrnet_st
                ensemble_row[f'ensemble_hrnetv2_{st_name}_600px'] = hrnet_st_scaled
                ensemble_row[f'ensemble_hrnetv2_{st_name}_error_224px'] = hrnet_st_error_224
                ensemble_row[f'ensemble_hrnetv2_{st_name}_error_600px'] = hrnet_st_error_600
                
                # Add mm measurements if calibration is available
                if mm_per_pixel_224:
                    hrnet_st_mm = hrnet_st * mm_per_pixel_224 if not np.isnan(hrnet_st) else np.nan
                    hrnet_st_error_mm = hrnet_st_error_224 * mm_per_pixel_224 if not np.isnan(hrnet_st_error_224) else np.nan
                    ensemble_row[f'ensemble_hrnetv2_{st_name}_mm'] = hrnet_st_mm
                    ensemble_row[f'ensemble_hrnetv2_{st_name}_error_mm'] = hrnet_st_error_mm
            else:
                hrnet_st_error = abs(hrnet_st - gt_st) if not np.isnan(gt_st) and not np.isnan(hrnet_st) else np.nan
                ensemble_row[f'ensemble_hrnetv2_{st_name}'] = hrnet_st
                ensemble_row[f'ensemble_hrnetv2_{st_name}_error'] = hrnet_st_error
            
            # Ensemble MLP
            mlp_st = ensemble_mlp_soft_tissue.get(st_name, np.nan)
            
            if 'eline' in st_name and not np.isnan(mlp_st):
                mlp_st_scaled = mlp_st * SCALE_FACTOR
                mlp_st_error_224 = abs(mlp_st - gt_st) if not np.isnan(gt_st) else np.nan
                mlp_st_error_600 = mlp_st_error_224 * SCALE_FACTOR if not np.isnan(mlp_st_error_224) else np.nan
                
                ensemble_row[f'ensemble_mlp_{st_name}_224px'] = mlp_st
                ensemble_row[f'ensemble_mlp_{st_name}_600px'] = mlp_st_scaled
                ensemble_row[f'ensemble_mlp_{st_name}_error_224px'] = mlp_st_error_224
                ensemble_row[f'ensemble_mlp_{st_name}_error_600px'] = mlp_st_error_600
                
                # Add mm measurements if calibration is available
                if mm_per_pixel_224:
                    mlp_st_mm = mlp_st * mm_per_pixel_224 if not np.isnan(mlp_st) else np.nan
                    mlp_st_error_mm = mlp_st_error_224 * mm_per_pixel_224 if not np.isnan(mlp_st_error_224) else np.nan
                    ensemble_row[f'ensemble_mlp_{st_name}_mm'] = mlp_st_mm
                    ensemble_row[f'ensemble_mlp_{st_name}_error_mm'] = mlp_st_error_mm
            else:
                mlp_st_error = abs(mlp_st - gt_st) if not np.isnan(gt_st) and not np.isnan(mlp_st) else np.nan
                ensemble_row[f'ensemble_mlp_{st_name}'] = mlp_st
                ensemble_row[f'ensemble_mlp_{st_name}_error'] = mlp_st_error
        
        # Add patient classification
        ensemble_row['gt_classification'] = gt_class
        ensemble_row['ensemble_hrnetv2_classification'] = ensemble_hrnet_class
        ensemble_row['ensemble_mlp_classification'] = ensemble_mlp_class
        
        ensemble_angle_data.append(ensemble_row)
        
        # Calculate individual model angles
        individual_row = {'patient_id': patient_id}
        
        # Add ground truth angles and soft tissue
        for angle_name in angle_names:
            individual_row[f'gt_{angle_name}'] = gt_angles.get(angle_name, np.nan)
        for st_name in soft_tissue_names:
            individual_row[f'gt_{st_name}'] = gt_soft_tissue.get(st_name, np.nan)
        individual_row['gt_classification'] = gt_class
        
        # Add individual model predictions
        for model_idx in range(len(all_hrnet_preds)):
            # HRNet model angles
            model_hrnet_angles = calculate_cephalometric_angles(all_hrnet_preds[model_idx][i], landmark_names)
            model_hrnet_soft_tissue = calculate_soft_tissue_measurements(all_hrnet_preds[model_idx][i], landmark_names)
            
            model_mlp_angles = calculate_cephalometric_angles(all_mlp_preds[model_idx][i], landmark_names)
            model_mlp_soft_tissue = calculate_soft_tissue_measurements(all_mlp_preds[model_idx][i], landmark_names)
            
            # Classifications
            model_hrnet_anb = model_hrnet_angles.get('ANB', np.nan)
            model_hrnet_class = classify_patient(model_hrnet_anb)
            individual_model_classifications[model_idx].append(model_hrnet_class)
            
            model_mlp_anb = model_mlp_angles.get('ANB', np.nan)
            model_mlp_class = classify_patient(model_mlp_anb)
            
            for angle_name in angle_names:
                gt_angle = gt_angles.get(angle_name, np.nan)
                
                # HRNet model
                hrnet_angle = model_hrnet_angles.get(angle_name, np.nan)
                hrnet_error = abs(hrnet_angle - gt_angle) if not np.isnan(gt_angle) and not np.isnan(hrnet_angle) else np.nan
                individual_row[f'model{model_idx+1}_hrnetv2_{angle_name}'] = hrnet_angle
                individual_row[f'model{model_idx+1}_hrnetv2_{angle_name}_error'] = hrnet_error
                
                # MLP model
                mlp_angle = model_mlp_angles.get(angle_name, np.nan)
                mlp_error = abs(mlp_angle - gt_angle) if not np.isnan(gt_angle) and not np.isnan(mlp_angle) else np.nan
                individual_row[f'model{model_idx+1}_mlp_{angle_name}'] = mlp_angle
                individual_row[f'model{model_idx+1}_mlp_{angle_name}_error'] = mlp_error
            
            # Add soft tissue for individual models
            for st_name in soft_tissue_names:
                gt_st = gt_soft_tissue.get(st_name, np.nan)
                
                # HRNet model
                hrnet_st = model_hrnet_soft_tissue.get(st_name, np.nan)
                
                if 'eline' in st_name and not np.isnan(hrnet_st):
                    hrnet_st_scaled = hrnet_st * SCALE_FACTOR
                    hrnet_st_error_224 = abs(hrnet_st - gt_st) if not np.isnan(gt_st) else np.nan
                    hrnet_st_error_600 = hrnet_st_error_224 * SCALE_FACTOR if not np.isnan(hrnet_st_error_224) else np.nan
                    
                    individual_row[f'model{model_idx+1}_hrnetv2_{st_name}_224px'] = hrnet_st
                    individual_row[f'model{model_idx+1}_hrnetv2_{st_name}_600px'] = hrnet_st_scaled
                    individual_row[f'model{model_idx+1}_hrnetv2_{st_name}_error_224px'] = hrnet_st_error_224
                    individual_row[f'model{model_idx+1}_hrnetv2_{st_name}_error_600px'] = hrnet_st_error_600
                else:
                    hrnet_st_error = abs(hrnet_st - gt_st) if not np.isnan(gt_st) and not np.isnan(hrnet_st) else np.nan
                    individual_row[f'model{model_idx+1}_hrnetv2_{st_name}'] = hrnet_st
                    individual_row[f'model{model_idx+1}_hrnetv2_{st_name}_error'] = hrnet_st_error
                
                # MLP model
                mlp_st = model_mlp_soft_tissue.get(st_name, np.nan)
                
                if 'eline' in st_name and not np.isnan(mlp_st):
                    mlp_st_scaled = mlp_st * SCALE_FACTOR
                    mlp_st_error_224 = abs(mlp_st - gt_st) if not np.isnan(gt_st) else np.nan
                    mlp_st_error_600 = mlp_st_error_224 * SCALE_FACTOR if not np.isnan(mlp_st_error_224) else np.nan
                    
                    individual_row[f'model{model_idx+1}_mlp_{st_name}_224px'] = mlp_st
                    individual_row[f'model{model_idx+1}_mlp_{st_name}_600px'] = mlp_st_scaled
                    individual_row[f'model{model_idx+1}_mlp_{st_name}_error_224px'] = mlp_st_error_224
                    individual_row[f'model{model_idx+1}_mlp_{st_name}_error_600px'] = mlp_st_error_600
                else:
                    mlp_st_error = abs(mlp_st - gt_st) if not np.isnan(gt_st) and not np.isnan(mlp_st) else np.nan
                    individual_row[f'model{model_idx+1}_mlp_{st_name}'] = mlp_st
                    individual_row[f'model{model_idx+1}_mlp_{st_name}_error'] = mlp_st_error
            
            # Add classifications
            individual_row[f'model{model_idx+1}_hrnetv2_classification'] = model_hrnet_class
            individual_row[f'model{model_idx+1}_mlp_classification'] = model_mlp_class
        
        # Add ensemble angles at the end
        for angle_name in angle_names:
            individual_row[f'ensemble_hrnetv2_{angle_name}'] = ensemble_hrnet_angles.get(angle_name, np.nan)
            individual_row[f'ensemble_hrnetv2_{angle_name}_error'] = ensemble_row[f'ensemble_hrnetv2_{angle_name}_error']
            individual_row[f'ensemble_mlp_{angle_name}'] = ensemble_mlp_angles.get(angle_name, np.nan)
            individual_row[f'ensemble_mlp_{angle_name}_error'] = ensemble_row[f'ensemble_mlp_{angle_name}_error']
        
        # Add ensemble soft tissue
        for st_name in soft_tissue_names:
            if 'eline' in st_name:
                individual_row[f'ensemble_hrnetv2_{st_name}_224px'] = ensemble_row.get(f'ensemble_hrnetv2_{st_name}_224px', np.nan)
                individual_row[f'ensemble_hrnetv2_{st_name}_600px'] = ensemble_row.get(f'ensemble_hrnetv2_{st_name}_600px', np.nan)
                individual_row[f'ensemble_hrnetv2_{st_name}_error_224px'] = ensemble_row.get(f'ensemble_hrnetv2_{st_name}_error_224px', np.nan)
                individual_row[f'ensemble_hrnetv2_{st_name}_error_600px'] = ensemble_row.get(f'ensemble_hrnetv2_{st_name}_error_600px', np.nan)
                
                individual_row[f'ensemble_mlp_{st_name}_224px'] = ensemble_row.get(f'ensemble_mlp_{st_name}_224px', np.nan)
                individual_row[f'ensemble_mlp_{st_name}_600px'] = ensemble_row.get(f'ensemble_mlp_{st_name}_600px', np.nan)
                individual_row[f'ensemble_mlp_{st_name}_error_224px'] = ensemble_row.get(f'ensemble_mlp_{st_name}_error_224px', np.nan)
                individual_row[f'ensemble_mlp_{st_name}_error_600px'] = ensemble_row.get(f'ensemble_mlp_{st_name}_error_600px', np.nan)
            else:
                individual_row[f'ensemble_hrnetv2_{st_name}'] = ensemble_hrnet_soft_tissue.get(st_name, np.nan)
                individual_row[f'ensemble_hrnetv2_{st_name}_error'] = ensemble_row.get(f'ensemble_hrnetv2_{st_name}_error', np.nan)
                individual_row[f'ensemble_mlp_{st_name}'] = ensemble_mlp_soft_tissue.get(st_name, np.nan)
                individual_row[f'ensemble_mlp_{st_name}_error'] = ensemble_row.get(f'ensemble_mlp_{st_name}_error', np.nan)
        
        # Add ensemble classifications
        individual_row['ensemble_hrnetv2_classification'] = ensemble_hrnet_class
        individual_row['ensemble_mlp_classification'] = ensemble_mlp_class
        
        individual_angle_data.append(individual_row)
    
    # Create DataFrames and save to CSV
    ensemble_angle_df = pd.DataFrame(ensemble_angle_data)
    individual_angle_df = pd.DataFrame(individual_angle_data)
    
    # Save files
    ensemble_angle_csv_path = os.path.join(output_dir, "ensemble_angle_predictions.csv")
    individual_angle_csv_path = os.path.join(output_dir, "all_models_angle_predictions.csv")
    
    ensemble_angle_df.to_csv(ensemble_angle_csv_path, index=False)
    individual_angle_df.to_csv(individual_angle_csv_path, index=False)
    
    print(f"   ✓ Ensemble angle predictions saved to: {os.path.basename(ensemble_angle_csv_path)}")
    print(f"   ✓ All models angle predictions saved to: {os.path.basename(individual_angle_csv_path)}")
    
    # Calculate and print angle error statistics
    print(f"\n📊 Angle Error Statistics:")
    print(f"{'Measurement':<20} {'GT Mean':<12} {'Ensemble MLP MAE':<20} {'Ensemble HRNet MAE':<20} {'Improvement':<15}")
    print("-" * 95)
    
    # Print angle statistics
    for angle_name in angle_names:
        # Get ground truth values
        gt_values = ensemble_angle_df[f'gt_{angle_name}'].dropna()
        
        if len(gt_values) > 0:
            gt_mean = gt_values.mean()
            
            # Get ensemble errors
            mlp_errors = ensemble_angle_df[f'ensemble_mlp_{angle_name}_error'].dropna()
            hrnet_errors = ensemble_angle_df[f'ensemble_hrnetv2_{angle_name}_error'].dropna()
            
            if len(mlp_errors) > 0 and len(hrnet_errors) > 0:
                mlp_mae = mlp_errors.mean()
                hrnet_mae = hrnet_errors.mean()
                improvement = (hrnet_mae - mlp_mae) / hrnet_mae * 100 if hrnet_mae > 0 else 0
                
                print(f"{angle_name:<20} {gt_mean:<12.1f} {mlp_mae:<20.2f} {hrnet_mae:<20.2f} {improvement:<15.1f}%")
    
    # Print soft tissue statistics
    print("\n📏 Soft Tissue Measurement Statistics:")
    for st_name in soft_tissue_names:
        # For E-line measurements, we need to calculate them from the coordinates
        if 'eline' in st_name:
            # Calculate ground truth E-line measurements
            gt_values = []
            mlp_errors = []
            hrnet_errors = []
            
            for i, patient_id in enumerate(patient_ids):
                # Calculate ground truth measurement
                gt_st = calculate_soft_tissue_measurements(gt_coords[i], landmark_names).get(st_name, np.nan)
                if not np.isnan(gt_st):
                    gt_values.append(gt_st)
                    
                    # Calculate ensemble predictions
                    mlp_st = calculate_soft_tissue_measurements(ensemble_mlp[i], landmark_names).get(st_name, np.nan)
                    hrnet_st = calculate_soft_tissue_measurements(ensemble_hrnet[i], landmark_names).get(st_name, np.nan)
                    
                    if not np.isnan(mlp_st):
                        mlp_errors.append(abs(mlp_st - gt_st))
                    if not np.isnan(hrnet_st):
                        hrnet_errors.append(abs(hrnet_st - gt_st))
            
            if gt_values:
                gt_mean = np.mean(gt_values)
                
                if mlp_errors and hrnet_errors:
                    mlp_mae = np.mean(mlp_errors)
                    hrnet_mae = np.mean(hrnet_errors)
                    improvement = (hrnet_mae - mlp_mae) / hrnet_mae * 100 if hrnet_mae > 0 else 0
                    
                    # Convert to mm if calibration is available
                    if ruler_data:
                        print(f"{st_name:<20} {gt_mean:<12.1f}px {mlp_mae:<20.2f} {hrnet_mae:<20.2f} {improvement:<15.1f}%")
                        
                        # Calculate mm statistics
                        gt_values_mm = []
                        mlp_errors_mm = []
                        hrnet_errors_mm = []
                        
                        for i, patient_id in enumerate(patient_ids):
                            mm_per_pixel_224 = None
                            mm_per_pixel_600 = calculate_pixel_to_mm_ratio(ruler_data, patient_id)
                            if mm_per_pixel_600:
                                mm_per_pixel_224 = mm_per_pixel_600 * SCALE_FACTOR
                                
                                gt_st = calculate_soft_tissue_measurements(gt_coords[i], landmark_names).get(st_name, np.nan)
                                if not np.isnan(gt_st):
                                    gt_values_mm.append(gt_st * mm_per_pixel_224)
                                    
                                    mlp_st = calculate_soft_tissue_measurements(ensemble_mlp[i], landmark_names).get(st_name, np.nan)
                                    hrnet_st = calculate_soft_tissue_measurements(ensemble_hrnet[i], landmark_names).get(st_name, np.nan)
                                    
                                    if not np.isnan(mlp_st):
                                        mlp_errors_mm.append(abs(mlp_st - gt_st) * mm_per_pixel_224)
                                    if not np.isnan(hrnet_st):
                                        hrnet_errors_mm.append(abs(hrnet_st - gt_st) * mm_per_pixel_224)
                        
                        if gt_values_mm:
                            gt_mean_mm = np.mean(gt_values_mm)
                            mlp_mae_mm = np.mean(mlp_errors_mm)
                            hrnet_mae_mm = np.mean(hrnet_errors_mm)
                            improvement_mm = (hrnet_mae_mm - mlp_mae_mm) / hrnet_mae_mm * 100 if hrnet_mae_mm > 0 else 0
                            print(f"   (in mm)            {gt_mean_mm:<12.1f}mm {mlp_mae_mm:<20.2f} {hrnet_mae_mm:<20.2f} {improvement_mm:<15.1f}%")
                    else:
                        print(f"{st_name:<20} {gt_mean:<12.1f}px {mlp_mae:<20.2f} {hrnet_mae:<20.2f} {improvement:<15.1f}%")
        else:
            # For non-E-line measurements (e.g., nasolabial angle)
            gt_values = ensemble_angle_df[f'gt_{st_name}'].dropna()
            
            if len(gt_values) > 0:
                gt_mean = gt_values.mean()
                
                # Get ensemble errors
                mlp_errors = ensemble_angle_df[f'ensemble_mlp_{st_name}_error'].dropna()
                hrnet_errors = ensemble_angle_df[f'ensemble_hrnetv2_{st_name}_error'].dropna()
                
                if len(mlp_errors) > 0 and len(hrnet_errors) > 0:
                    mlp_mae = mlp_errors.mean()
                    hrnet_mae = hrnet_errors.mean()
                    improvement = (hrnet_mae - mlp_mae) / hrnet_mae * 100 if hrnet_mae > 0 else 0
                    
                    unit = '°'
                    print(f"{st_name:<20} {gt_mean:<12.1f}{unit} {mlp_mae:<20.2f} {hrnet_mae:<20.2f} {improvement:<15.1f}%")
                
                # Also print mm errors for E-line measurements if available
                if 'eline' in st_name and ruler_data:
                    mlp_errors_mm = ensemble_angle_df[f'ensemble_mlp_{st_name}_error_mm'].dropna()
                    hrnet_errors_mm = ensemble_angle_df[f'ensemble_hrnetv2_{st_name}_error_mm'].dropna()
                    
                    if len(mlp_errors_mm) > 0 and len(hrnet_errors_mm) > 0:
                        mlp_mae_mm = mlp_errors_mm.mean()
                        hrnet_mae_mm = hrnet_errors_mm.mean()
                        improvement_mm = (hrnet_mae_mm - mlp_mae_mm) / hrnet_mae_mm * 100 if hrnet_mae_mm > 0 else 0
                        print(f"   (in mm)            {'':<12} {mlp_mae_mm:<20.2f} {hrnet_mae_mm:<20.2f} {improvement_mm:<15.1f}%")
    
    # Calculate and print classification metrics
    print(f"\n🏷️  Patient Classification Metrics (based on ANB angle):")
    print("-" * 70)
    
    # Ensemble HRNetV2 classification metrics
    hrnet_metrics = calculate_classification_metrics(gt_classifications, ensemble_hrnet_classifications)
    print(f"\nEnsemble HRNetV2 Classification:")
    print(f"  Accuracy: {hrnet_metrics['accuracy']:.3f}")
    print(f"  Precision (macro): {hrnet_metrics['precision']:.3f}")
    print(f"  Recall (macro): {hrnet_metrics['recall']:.3f}")
    print(f"  F1-Score (macro): {hrnet_metrics['f1_score']:.3f}")
    print(f"  Valid samples: {hrnet_metrics['n_samples']}")
    
    # Ensemble MLP classification metrics
    mlp_metrics = calculate_classification_metrics(gt_classifications, ensemble_mlp_classifications)
    print(f"\nEnsemble MLP Classification:")
    print(f"  Accuracy: {mlp_metrics['accuracy']:.3f}")
    print(f"  Precision (macro): {mlp_metrics['precision']:.3f}")
    print(f"  Recall (macro): {mlp_metrics['recall']:.3f}")
    print(f"  F1-Score (macro): {mlp_metrics['f1_score']:.3f}")
    print(f"  Valid samples: {mlp_metrics['n_samples']}")
    
    # Save classification metrics
    classification_results = {
        'ensemble_hrnetv2': hrnet_metrics,
        'ensemble_mlp': mlp_metrics
    }
    
    # Create comprehensive visualization
    create_angle_error_visualization(ensemble_angle_df, angle_names, soft_tissue_names, output_dir)
    create_classification_visualization(classification_results, output_dir)
    
    # Return classification results for the overall report
    return classification_results

def create_angle_error_visualization(angle_df: pd.DataFrame, angle_names: List[str], 
                                   soft_tissue_names: List[str], output_dir: str):
    """Create visualization of angle and soft tissue measurement errors."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Average angle errors comparison
    ax = axes[0, 0]
    
    mlp_errors = []
    hrnet_errors = []
    
    for angle_name in angle_names:
        mlp_err = angle_df[f'ensemble_mlp_{angle_name}_error'].dropna().mean()
        hrnet_err = angle_df[f'ensemble_hrnetv2_{angle_name}_error'].dropna().mean()
        mlp_errors.append(mlp_err)
        hrnet_errors.append(hrnet_err)
    
    x = np.arange(len(angle_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, hrnet_errors, width, label='Ensemble HRNetV2', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, mlp_errors, width, label='Ensemble MLP', color='red', alpha=0.7)
    
    ax.set_xlabel('Angle')
    ax.set_ylabel('Mean Absolute Error (degrees)')
    ax.set_title('Cephalometric Angle Errors')
    ax.set_xticks(x)
    ax.set_xticklabels(angle_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add improvement percentage on top
    for i, (h_err, m_err) in enumerate(zip(hrnet_errors, mlp_errors)):
        if h_err > 0:
            improvement = (h_err - m_err) / h_err * 100
            color = 'green' if improvement > 0 else 'red'
            ax.text(i, max(h_err, m_err) + 0.1, f'{improvement:+.0f}%', 
                   ha='center', va='bottom', fontsize=8, color=color)
    
    # Plot 2: Error distribution for each angle
    ax = axes[0, 1]
    
    # Box plot of errors
    mlp_error_data = []
    hrnet_error_data = []
    labels = []
    
    for angle_name in angle_names:
        mlp_err = angle_df[f'ensemble_mlp_{angle_name}_error'].dropna()
        hrnet_err = angle_df[f'ensemble_hrnetv2_{angle_name}_error'].dropna()
        
        if len(mlp_err) > 0:
            mlp_error_data.append(mlp_err)
            hrnet_error_data.append(hrnet_err)
            labels.append(angle_name)
    
    positions = np.arange(len(labels))
    bp1 = ax.boxplot(hrnet_error_data, positions=positions - 0.2, widths=0.35, 
                     patch_artist=True, boxprops=dict(facecolor='lightblue'))
    bp2 = ax.boxplot(mlp_error_data, positions=positions + 0.2, widths=0.35, 
                     patch_artist=True, boxprops=dict(facecolor='lightcoral'))
    
    ax.set_xlabel('Angle')
    ax.set_ylabel('Error Distribution (degrees)')
    ax.set_title('Angle Error Distributions')
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Ensemble HRNetV2', 'Ensemble MLP'])
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Patient-wise angle errors
    ax = axes[1, 0]
    
    # Calculate total angle error per patient
    patient_mlp_errors = []
    patient_hrnet_errors = []
    
    for _, row in angle_df.iterrows():
        mlp_total = 0
        hrnet_total = 0
        count = 0
        
        for angle_name in angle_names:
            mlp_err = row[f'ensemble_mlp_{angle_name}_error']
            hrnet_err = row[f'ensemble_hrnetv2_{angle_name}_error']
            
            if not np.isnan(mlp_err) and not np.isnan(hrnet_err):
                mlp_total += mlp_err
                hrnet_total += hrnet_err
                count += 1
        
        if count > 0:
            patient_mlp_errors.append(mlp_total / count)
            patient_hrnet_errors.append(hrnet_total / count)
    
    ax.scatter(patient_hrnet_errors, patient_mlp_errors, alpha=0.6, color='purple')
    
    # Add diagonal line
    max_error = max(max(patient_hrnet_errors), max(patient_mlp_errors))
    ax.plot([0, max_error], [0, max_error], 'k--', alpha=0.5, label='Equal error line')
    
    # Add improvement region
    ax.fill_between([0, max_error], [0, max_error], [0, 0], alpha=0.1, color='green', 
                    label='MLP improvement region')
    
    ax.set_xlabel('Ensemble HRNetV2 Average Angle Error (degrees)')
    ax.set_ylabel('Ensemble MLP Average Angle Error (degrees)')
    ax.set_title('Patient-wise Angle Error Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Angle importance (by error magnitude)
    ax = axes[1, 1]
    
    # Sort angles by average error
    angle_importance = []
    for i, angle_name in enumerate(angle_names):
        avg_error = (mlp_errors[i] + hrnet_errors[i]) / 2
        improvement = (hrnet_errors[i] - mlp_errors[i]) / hrnet_errors[i] * 100 if hrnet_errors[i] > 0 else 0
        angle_importance.append((angle_name, avg_error, improvement))
    
    angle_importance.sort(key=lambda x: x[1], reverse=True)
    
    sorted_names = [x[0] for x in angle_importance]
    sorted_errors = [x[1] for x in angle_importance]
    sorted_improvements = [x[2] for x in angle_importance]
    
    bars = ax.barh(range(len(sorted_names)), sorted_errors, color='gray', alpha=0.7)
    
    # Color bars by improvement
    for i, (bar, imp) in enumerate(zip(bars, sorted_improvements)):
        if imp > 10:
            bar.set_color('green')
            bar.set_alpha(0.7)
        elif imp < -5:
            bar.set_color('red')
            bar.set_alpha(0.7)
    
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('Average Error (degrees)')
    ax.set_title('Angle Measurement Difficulty\n(Green: MLP improves >10%, Red: MLP worse)')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Cephalometric Angle Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'angle_error_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Angle error visualization saved to: {os.path.basename(output_path)}")

def calculate_per_patient_errors(pred_coords: np.ndarray, gt_coords: np.ndarray, patient_ids: List[int]) -> List[Tuple[int, float]]:
    """Calculate average error per patient and return sorted list of (patient_id, error)."""
    patient_errors = []
    
    for i, patient_id in enumerate(patient_ids):
        # Calculate errors for all landmarks of this patient
        valid_mask = (gt_coords[i, :, 0] > 0) & (gt_coords[i, :, 1] > 0)
        if np.any(valid_mask):
            errors = np.sqrt(np.sum((pred_coords[i, valid_mask] - gt_coords[i, valid_mask])**2, axis=1))
            avg_error = np.mean(errors)
            patient_errors.append((patient_id, avg_error))
    
    # Sort by error (ascending)
    patient_errors.sort(key=lambda x: x[1])
    return patient_errors

def visualize_patient_predictions(patient_idx: int, patient_id: int, 
                                 gt_coords: np.ndarray, ensemble_hrnet: np.ndarray, ensemble_mlp: np.ndarray,
                                 all_hrnet_preds: List[np.ndarray], all_mlp_preds: List[np.ndarray],
                                 landmark_names: List[str], image_data: Optional[np.ndarray],
                                 output_path: str, title_suffix: str = ""):
    """Create visualization for a single patient showing GT and predictions."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    # Define colors
    gt_color = 'green'
    ensemble_hrnet_color = 'blue'
    ensemble_mlp_color = 'red'
    individual_model_colors = ['cyan', 'magenta', 'yellow', 'orange', 'purple']
    
    # Get patient data
    gt = gt_coords[patient_idx]
    ens_hrnet = ensemble_hrnet[patient_idx]
    ens_mlp = ensemble_mlp[patient_idx]
    
    # Calculate errors
    valid_mask = (gt[:, 0] > 0) & (gt[:, 1] > 0)
    hrnet_errors = np.sqrt(np.sum((ens_hrnet[valid_mask] - gt[valid_mask])**2, axis=1))
    mlp_errors = np.sqrt(np.sum((ens_mlp[valid_mask] - gt[valid_mask])**2, axis=1))
    avg_hrnet_error = np.mean(hrnet_errors)
    avg_mlp_error = np.mean(mlp_errors)
    
    # Plot 1: Ground Truth vs Ensemble HRNetV2
    ax = axes[0]
    if image_data is not None:
        ax.imshow(image_data, cmap='gray')
    
    # Plot points
    for i, (g, p) in enumerate(zip(gt[valid_mask], ens_hrnet[valid_mask])):
        ax.scatter(g[0], g[1], c=gt_color, s=50, marker='o', alpha=0.8, edgecolors='black', linewidth=1)
        ax.scatter(p[0], p[1], c=ensemble_hrnet_color, s=30, marker='^', alpha=0.8)
        # Draw line between GT and prediction
        ax.plot([g[0], p[0]], [g[1], p[1]], 'gray', alpha=0.3, linewidth=0.5)
    
    ax.set_title(f'Patient {patient_id}: GT vs Ensemble HRNetV2\nMean Error: {avg_hrnet_error:.2f} pixels', fontsize=12)
    ax.axis('equal')
    ax.set_xlim(0, 224)
    ax.set_ylim(224, 0)  # Invert y-axis for image coordinates
    
    # Plot 2: Ground Truth vs Ensemble MLP
    ax = axes[1]
    if image_data is not None:
        ax.imshow(image_data, cmap='gray')
    
    for i, (g, p) in enumerate(zip(gt[valid_mask], ens_mlp[valid_mask])):
        ax.scatter(g[0], g[1], c=gt_color, s=50, marker='o', alpha=0.8, edgecolors='black', linewidth=1)
        ax.scatter(p[0], p[1], c=ensemble_mlp_color, s=30, marker='s', alpha=0.8)
        ax.plot([g[0], p[0]], [g[1], p[1]], 'gray', alpha=0.3, linewidth=0.5)
    
    ax.set_title(f'Patient {patient_id}: GT vs Ensemble MLP\nMean Error: {avg_mlp_error:.2f} pixels', fontsize=12)
    ax.axis('equal')
    ax.set_xlim(0, 224)
    ax.set_ylim(224, 0)
    
    # Plot 3: All Models Comparison
    ax = axes[2]
    if image_data is not None:
        ax.imshow(image_data, cmap='gray')
    
    # Plot GT
    ax.scatter(gt[valid_mask, 0], gt[valid_mask, 1], c=gt_color, s=100, marker='o', 
               alpha=0.8, edgecolors='black', linewidth=2, label='Ground Truth')
    
    # Plot individual models
    for model_idx in range(len(all_mlp_preds)):
        model_pred = all_mlp_preds[model_idx][patient_idx]
        color = individual_model_colors[model_idx % len(individual_model_colors)]
        ax.scatter(model_pred[valid_mask, 0], model_pred[valid_mask, 1], 
                  c=color, s=30, marker='x', alpha=0.6, label=f'Model {model_idx+1}')
    
    # Plot ensemble
    ax.scatter(ens_mlp[valid_mask, 0], ens_mlp[valid_mask, 1], 
              c=ensemble_mlp_color, s=50, marker='s', alpha=0.8, label='Ensemble MLP')
    
    ax.set_title(f'Patient {patient_id}: All Models Comparison', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.axis('equal')
    ax.set_xlim(0, 224)
    ax.set_ylim(224, 0)
    
    # Plot 4: Per-Landmark Errors
    ax = axes[3]
    
    # Get landmark names for valid landmarks
    valid_landmark_names = [landmark_names[i] for i in range(len(landmark_names)) if valid_mask[i]]
    hrnet_landmark_errors = hrnet_errors
    mlp_landmark_errors = mlp_errors
    
    x = np.arange(len(valid_landmark_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, hrnet_landmark_errors, width, label='Ensemble HRNetV2', color=ensemble_hrnet_color, alpha=0.7)
    bars2 = ax.bar(x + width/2, mlp_landmark_errors, width, label='Ensemble MLP', color=ensemble_mlp_color, alpha=0.7)
    
    ax.set_xlabel('Landmark')
    ax.set_ylabel('Error (pixels)')
    ax.set_title(f'Patient {patient_id}: Per-Landmark Errors')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_landmark_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add improvement percentage on top of bars
    for i, (h_err, m_err) in enumerate(zip(hrnet_landmark_errors, mlp_landmark_errors)):
        if h_err > 0:
            improvement = (h_err - m_err) / h_err * 100
            color = 'green' if improvement > 0 else 'red'
            ax.text(i, max(h_err, m_err) + 0.5, f'{improvement:+.0f}%', 
                   ha='center', va='bottom', fontsize=8, color=color)
    
    plt.suptitle(f'Patient {patient_id} {title_suffix}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved visualization: {os.path.basename(output_path)}")

def save_patient_json_data(patient_idx: int, patient_id: int, 
                          ensemble_mlp: np.ndarray, gt_coords: np.ndarray,
                          landmark_names: List[str], output_path: str,
                          ruler_data: Optional[Dict[str, Dict]] = None):
    """Save patient data in JSON format with the specified structure."""
    
    # Get patient data
    pred_coords = ensemble_mlp[patient_idx]  # Shape: [19, 2]
    gt_patient_coords = gt_coords[patient_idx]  # Shape: [19, 2]
    
    # Calculate angles and soft tissue measurements
    pred_angles = calculate_cephalometric_angles(pred_coords, landmark_names)
    pred_soft_tissue = calculate_soft_tissue_measurements(pred_coords, landmark_names)
    
    # Get patient classification
    anb_angle = pred_angles.get('ANB', np.nan)
    classification = classify_patient(anb_angle)
    
    # Calculate pixel to mm ratio for this patient
    mm_per_pixel_600 = None
    if ruler_data:
        mm_per_pixel_600 = calculate_pixel_to_mm_ratio(ruler_data, patient_id)
    
    # Create landmarks dictionary (scaled to 600x600)
    landmarks = {}
    groundtruth_landmarks = {}
    
    for i, landmark_name in enumerate(landmark_names):
        # Predicted landmarks (scale to 600x600)
        pred_x = pred_coords[i, 0] * SCALE_FACTOR
        pred_y = pred_coords[i, 1] * SCALE_FACTOR
        
        # Ground truth landmarks (scale to 600x600)
        gt_x = gt_patient_coords[i, 0] * SCALE_FACTOR if gt_patient_coords[i, 0] > 0 else 0
        gt_y = gt_patient_coords[i, 1] * SCALE_FACTOR if gt_patient_coords[i, 1] > 0 else 0
        
        # Handle missing landmarks
        if gt_patient_coords[i, 0] > 0 and gt_patient_coords[i, 1] > 0:
            landmarks[landmark_name] = {"x": float(pred_x), "y": float(pred_y)}
            groundtruth_landmarks[landmark_name] = {"x": float(gt_x), "y": float(gt_y)}
        else:
            landmarks[landmark_name] = None
            groundtruth_landmarks[landmark_name] = None
    
    # Create angles dictionary (filter out NaN values)
    angles = {}
    for angle_name, angle_value in pred_angles.items():
        if not np.isnan(angle_value):
            angles[angle_name] = float(angle_value)
    
    # Create soft tissue dictionary
    soft_tissue = {}
    nasolabial = pred_soft_tissue.get('nasolabial_angle', np.nan)
    if not np.isnan(nasolabial):
        soft_tissue['nasolabial_angle'] = float(nasolabial)
    
    # Create distances dictionary
    distances = {}
    
    # Upper lip to E-line
    upper_lip_dist_224 = pred_soft_tissue.get('upper_lip_to_eline', np.nan)
    if not np.isnan(upper_lip_dist_224):
        upper_lip_dist_600 = upper_lip_dist_224 * SCALE_FACTOR
        upper_lip_entry = {
            "pixels_224": float(upper_lip_dist_224),
            "pixels_600": float(upper_lip_dist_600)
        }
        
        # Add mm if calibration is available
        if mm_per_pixel_600:
            upper_lip_dist_mm = upper_lip_dist_600 * mm_per_pixel_600
            upper_lip_entry["mm"] = float(upper_lip_dist_mm)
        
        distances["upper_lip_to_eline"] = upper_lip_entry
    
    # Lower lip to E-line
    lower_lip_dist_224 = pred_soft_tissue.get('lower_lip_to_eline', np.nan)
    if not np.isnan(lower_lip_dist_224):
        lower_lip_dist_600 = lower_lip_dist_224 * SCALE_FACTOR
        lower_lip_entry = {
            "pixels_224": float(lower_lip_dist_224),
            "pixels_600": float(lower_lip_dist_600)
        }
        
        # Add mm if calibration is available
        if mm_per_pixel_600:
            lower_lip_dist_mm = lower_lip_dist_600 * mm_per_pixel_600
            lower_lip_entry["mm"] = float(lower_lip_dist_mm)
        
        distances["lower_lip_to_eline"] = lower_lip_entry
    
    # Create the complete patient data structure
    patient_data = {
        "id": int(patient_id),
        "classification": classification,
        "landmarks": landmarks,
        "groundtruth_landmarks": groundtruth_landmarks,
        "angles": angles,
        "soft_tissue": soft_tissue,
        "distances": distances
    }
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(patient_data, f, indent=2)
    
    return patient_data

def create_patient_visualizations(ensemble_hrnet: np.ndarray, ensemble_mlp: np.ndarray,
                                all_hrnet_preds: List[np.ndarray], all_mlp_preds: List[np.ndarray],
                                gt_coords: np.ndarray, patient_ids: List[int],
                                test_df: pd.DataFrame, landmark_names: List[str], 
                                output_dir: str, ruler_data: Optional[Dict[str, Dict]] = None):
    """Create visualizations and JSON data for best and worst performing patients."""
    print(f"\n🎨 Creating patient visualizations and JSON data...")
    
    # Create visualization directory
    viz_dir = os.path.join(output_dir, "patient_visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Calculate per-patient errors for ensemble MLP
    patient_errors = calculate_per_patient_errors(ensemble_mlp, gt_coords, patient_ids)
    
    # Get best 5 and worst 3 patients
    best_patients = patient_errors[:5]  # First 5 (lowest errors)
    worst_patients = patient_errors[-3:]  # Last 3 (highest errors)
    
    print(f"\n📊 Best 5 patients (lowest average error):")
    for i, (pid, error) in enumerate(best_patients, 1):
        print(f"   {i}. Patient {pid}: {error:.2f} pixels")
    
    print(f"\n📊 Worst 3 patients (highest average error):")
    for i, (pid, error) in enumerate(worst_patients, 1):
        print(f"   {i}. Patient {pid}: {error:.2f} pixels")
    
    # Create visualizations and JSON data for best patients
    print(f"\n🎨 Creating visualizations and JSON data for best patients...")
    best_patients_data = []
    for rank, (patient_id, error) in enumerate(best_patients, 1):
        # Find patient index
        patient_idx = patient_ids.index(patient_id)
        
        # Try to get image data
        image_data = None
        try:
            # Find the row in test_df for this patient
            patient_row = test_df[test_df['patient_id'] == patient_id].iloc[0]
            if 'Image' in patient_row:
                image_data = np.array(patient_row['Image'], dtype=np.uint8).reshape((224, 224, 3))
        except:
            pass
        
        # Create visualization
        viz_output_path = os.path.join(viz_dir, f"best_{rank}_patient_{patient_id}.png")
        visualize_patient_predictions(
            patient_idx, patient_id, gt_coords, ensemble_hrnet, ensemble_mlp,
            all_hrnet_preds, all_mlp_preds, landmark_names, image_data,
            viz_output_path, f"(Best #{rank}, Avg Error: {error:.2f} pixels)"
        )
        
        # Create JSON data
        json_output_path = os.path.join(viz_dir, f"best_{rank}_patient_{patient_id}.json")
        patient_data = save_patient_json_data(
            patient_idx, patient_id, ensemble_mlp, gt_coords, 
            landmark_names, json_output_path, ruler_data
        )
        patient_data['rank'] = rank
        patient_data['category'] = 'best'
        patient_data['average_error_pixels'] = float(error)
        best_patients_data.append(patient_data)
        
        print(f"   ✓ Best #{rank} - Patient {patient_id}: visualization and JSON saved")
    
    # Create visualizations and JSON data for worst patients
    print(f"\n🎨 Creating visualizations and JSON data for worst patients...")
    worst_patients_data = []
    for rank, (patient_id, error) in enumerate(worst_patients, 1):
        # Find patient index
        patient_idx = patient_ids.index(patient_id)
        
        # Try to get image data
        image_data = None
        try:
            patient_row = test_df[test_df['patient_id'] == patient_id].iloc[0]
            if 'Image' in patient_row:
                image_data = np.array(patient_row['Image'], dtype=np.uint8).reshape((224, 224, 3))
        except:
            pass
        
        # Create visualization
        viz_output_path = os.path.join(viz_dir, f"worst_{rank}_patient_{patient_id}.png")
        visualize_patient_predictions(
            patient_idx, patient_id, gt_coords, ensemble_hrnet, ensemble_mlp,
            all_hrnet_preds, all_mlp_preds, landmark_names, image_data,
            viz_output_path, f"(Worst #{rank}, Avg Error: {error:.2f} pixels)"
        )
        
        # Create JSON data
        json_output_path = os.path.join(viz_dir, f"worst_{rank}_patient_{patient_id}.json")
        patient_data = save_patient_json_data(
            patient_idx, patient_id, ensemble_mlp, gt_coords, 
            landmark_names, json_output_path, ruler_data
        )
        patient_data['rank'] = rank
        patient_data['category'] = 'worst'
        patient_data['average_error_pixels'] = float(error)
        worst_patients_data.append(patient_data)
        
        print(f"   ✓ Worst #{rank} - Patient {patient_id}: visualization and JSON saved")
    
    # Save combined JSON file with all best and worst patients
    combined_data = {
        "evaluation_summary": {
            "total_patients_evaluated": len(patient_ids),
            "best_patients_count": len(best_patients),
            "worst_patients_count": len(worst_patients),
            "best_patients": best_patients_data,
            "worst_patients": worst_patients_data
        }
    }
    
    combined_json_path = os.path.join(viz_dir, "best_worst_patients_summary.json")
    with open(combined_json_path, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"\n💾 Combined summary saved to: {os.path.basename(combined_json_path)}")
    
    # Create summary visualization
    create_summary_visualization(patient_errors, ensemble_hrnet, ensemble_mlp, 
                               gt_coords, patient_ids, landmark_names, viz_dir)

def create_summary_visualization(patient_errors: List[Tuple[int, float]], 
                               ensemble_hrnet: np.ndarray, ensemble_mlp: np.ndarray,
                               gt_coords: np.ndarray, patient_ids: List[int],
                               landmark_names: List[str], output_dir: str):
    """Create summary visualization of overall performance."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Patient error distribution
    ax = axes[0, 0]
    errors = [err for _, err in patient_errors]
    ax.hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
    ax.axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.2f}')
    ax.set_xlabel('Average Error per Patient (pixels)')
    ax.set_ylabel('Number of Patients')
    ax.set_title('Patient Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Per-landmark performance comparison
    ax = axes[0, 1]
    
    # Calculate per-landmark errors for both methods
    landmark_errors_hrnet = []
    landmark_errors_mlp = []
    
    for j, landmark in enumerate(landmark_names):
        valid_mask = (gt_coords[:, j, 0] > 0) & (gt_coords[:, j, 1] > 0)
        if np.any(valid_mask):
            hrnet_errors = np.sqrt(np.sum((ensemble_hrnet[valid_mask, j] - gt_coords[valid_mask, j])**2, axis=1))
            mlp_errors = np.sqrt(np.sum((ensemble_mlp[valid_mask, j] - gt_coords[valid_mask, j])**2, axis=1))
            landmark_errors_hrnet.append(np.mean(hrnet_errors))
            landmark_errors_mlp.append(np.mean(mlp_errors))
        else:
            landmark_errors_hrnet.append(0)
            landmark_errors_mlp.append(0)
    
    # Sort landmarks by MLP error for better visualization
    sorted_indices = np.argsort(landmark_errors_mlp)[::-1]
    sorted_landmarks = [landmark_names[i] for i in sorted_indices]
    sorted_hrnet = [landmark_errors_hrnet[i] for i in sorted_indices]
    sorted_mlp = [landmark_errors_mlp[i] for i in sorted_indices]
    
    x = np.arange(len(sorted_landmarks))
    width = 0.35
    
    bars1 = ax.barh(x - width/2, sorted_hrnet, width, label='Ensemble HRNetV2', color='blue', alpha=0.7)
    bars2 = ax.barh(x + width/2, sorted_mlp, width, label='Ensemble MLP', color='red', alpha=0.7)
    
    ax.set_ylabel('Landmark')
    ax.set_xlabel('Mean Error (pixels)')
    ax.set_title('Per-Landmark Error Comparison')
    ax.set_yticks(x)
    ax.set_yticklabels(sorted_landmarks)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Improvement analysis
    ax = axes[1, 0]
    improvements = [(h - m) / h * 100 if h > 0 else 0 
                   for h, m in zip(landmark_errors_hrnet, landmark_errors_mlp)]
    
    sorted_improvements = [(landmark_names[i], improvements[i]) for i in range(len(improvements))]
    sorted_improvements.sort(key=lambda x: x[1], reverse=True)
    
    improvement_names = [x[0] for x in sorted_improvements]
    improvement_values = [x[1] for x in sorted_improvements]
    
    colors = ['green' if x > 0 else 'red' for x in improvement_values]
    bars = ax.bar(range(len(improvement_names)), improvement_values, color=colors, alpha=0.7)
    
    ax.set_xlabel('Landmark')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('MLP Improvement over HRNetV2 by Landmark')
    ax.set_xticks(range(len(improvement_names)))
    ax.set_xticklabels(improvement_names, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Best vs Worst patients comparison
    ax = axes[1, 1]
    
    best_errors = [err for _, err in patient_errors[:10]]
    worst_errors = [err for _, err in patient_errors[-10:]]
    
    positions = [1, 2]
    bp = ax.boxplot([best_errors, worst_errors], positions=positions, widths=0.6,
                    patch_artist=True, labels=['Best 10', 'Worst 10'])
    
    for patch, color in zip(bp['boxes'], ['lightgreen', 'lightcoral']):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Average Error (pixels)')
    ax.set_title('Best vs Worst Patients Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Ensemble Model Performance Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'performance_summary.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved summary visualization: {os.path.basename(output_path)}")

def print_results_table(results: Dict[str, Dict], landmark_names: List[str]):
    """Print formatted results table."""
    print(f"\n{'='*100}")
    print(f"📊 ENSEMBLE EVALUATION RESULTS")
    print(f"{'='*100}")
    
    # Overall performance table
    print(f"\n🏷️  OVERALL PERFORMANCE:")
    
    # Check if mm metrics are available
    has_mm_metrics = any('mre_mm' in metrics['overall'] for metrics in results.values())
    
    if has_mm_metrics:
        header = f"{'Model':<20} {'MRE (px)':<10} {'MRE (mm)':<10} {'Std (px)':<10} {'Std (mm)':<10} {'Median (px)':<12} {'Median (mm)':<12} {'Samples':<10}"
    else:
        header = f"{'Model':<20} {'MRE':<10} {'Std':<10} {'Median':<10} {'P90':<10} {'P95':<10} {'Samples':<10}"
    print(header)
    print("-" * len(header))
    
    for model_name, metrics in results.items():
        overall = metrics['overall']
        if has_mm_metrics and 'mre_mm' in overall:
            print(f"{model_name:<20} {overall['mre']:<10.3f} {overall['mre_mm']:<10.3f} "
                  f"{overall['std']:<10.3f} {overall['std_mm']:<10.3f} "
                  f"{overall['median']:<12.3f} {overall['median_mm']:<12.3f} "
                  f"{overall['count']:<10}")
        else:
            print(f"{model_name:<20} {overall['mre']:<10.3f} {overall['std']:<10.3f} "
                  f"{overall['median']:<10.3f} {overall['p90']:<10.3f} {overall['p95']:<10.3f} "
                  f"{overall['count']:<10}")
    
    # Key landmarks performance
    key_landmarks = ['sella', 'Gonion', 'PNS', 'A_point', 'B_point', 'ANS', 'nasion']
    available_landmarks = [lm for lm in key_landmarks if lm in landmark_names]
    
    print(f"\n🎯 KEY LANDMARKS PERFORMANCE:")
    
    for landmark in available_landmarks:
        print(f"\n{landmark.upper()}:")
        
        # Check if mm metrics are available for this landmark
        has_landmark_mm = any(landmark in metrics['per_landmark'] and 'mre_mm' in metrics['per_landmark'][landmark] 
                             for metrics in results.values())
        
        if has_landmark_mm:
            header = f"{'Model':<20} {'MRE (px)':<10} {'MRE (mm)':<10} {'Std (px)':<10} {'Std (mm)':<10} {'Count':<10}"
        else:
            header = f"{'Model':<20} {'MRE':<10} {'Std':<10} {'Median':<10} {'Count':<10}"
        print(header)
        print("-" * len(header))
        
        for model_name, metrics in results.items():
            if landmark in metrics['per_landmark']:
                lm_metrics = metrics['per_landmark'][landmark]
                if has_landmark_mm and 'mre_mm' in lm_metrics:
                    print(f"{model_name:<20} {lm_metrics['mre']:<10.3f} {lm_metrics['mre_mm']:<10.3f} "
                          f"{lm_metrics['std']:<10.3f} {lm_metrics['std_mm']:<10.3f} "
                          f"{lm_metrics['count']:<10}")
                else:
                    print(f"{model_name:<20} {lm_metrics['mre']:<10.3f} {lm_metrics['std']:<10.3f} "
                          f"{lm_metrics['median']:<10.3f} {lm_metrics['count']:<10}")
    
    # Improvement analysis (compare with first individual model)
    if len(results) > 2:  # At least 2 individual models + ensemble
        individual_models = [k for k in results.keys() if k.startswith('Model')]
        if individual_models and 'Ensemble MLP (Test)' in results:
            baseline_model = individual_models[0] + ' (Test)'
            if baseline_model in results:
                baseline_mre = results[baseline_model]['overall']['mre']
                ensemble_mre = results['Ensemble MLP (Test)']['overall']['mre']
                
                improvement = (baseline_mre - ensemble_mre) / baseline_mre * 100
                
                print(f"\n📈 ENSEMBLE IMPROVEMENT:")
                print(f"   Baseline ({baseline_model}): {baseline_mre:.3f} pixels")
                print(f"   Ensemble MLP (Test): {ensemble_mre:.3f} pixels")
                print(f"   Improvement: {improvement:+.1f}%")

def save_overall_results_report(results: Dict[str, Dict], validation_results: Dict[str, Dict],
                               classification_results: Dict[str, Dict], 
                               landmark_names: List[str], args: argparse.Namespace,
                               output_dir: str, n_test_samples: int, n_models_evaluated: int,
                               ruler_data: Optional[Dict[str, Dict]] = None):
    """Save a comprehensive overall results report to a text file."""
    report_path = os.path.join(output_dir, "overall_results_report.txt")
    
    with open(report_path, 'w') as f:
        # Header
        f.write("="*80 + "\n")
        f.write("ENSEMBLE CONCURRENT JOINT MLP EVALUATION - OVERALL RESULTS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Evaluation Configuration
        f.write("EVALUATION CONFIGURATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Base Work Directory: {args.base_work_dir}\n")
        f.write(f"Number of Models in Ensemble: {args.n_models}\n")
        f.write(f"Models Successfully Evaluated: {n_models_evaluated}\n")
        f.write(f"Evaluation Mode: {'Epoch ' + str(args.epoch) if args.epoch else 'Best/Latest Checkpoints'}\n")
        f.write(f"Test Samples: {n_test_samples}\n")
        f.write(f"Evaluate Individual Models: {args.evaluate_individual}\n")
        f.write(f"Evaluate on Validation: {args.evaluate_on_validation}\n")
        f.write(f"Image Scaling: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE} → {ORIGINAL_IMAGE_SIZE}x{ORIGINAL_IMAGE_SIZE} (factor: {SCALE_FACTOR:.4f})\n")
        if ruler_data:
            f.write(f"Millimeter Calibration: Available ({RULER_LENGTH_MM}mm ruler)\n")
        f.write("\n")
        
        # Overall Test Set Performance
        f.write("OVERALL TEST SET PERFORMANCE:\n")
        f.write("-" * 30 + "\n")
        
        # Check if mm metrics are available
        has_mm_metrics = any('mre_mm' in metrics['overall'] for metrics in results.values())
        
        # Header for metrics table
        if has_mm_metrics:
            header = f"{'Model':<25} {'MRE (px)':<12} {'MRE (mm)':<12} {'Std (px)':<12} {'Std (mm)':<12} {'Median (px)':<12} {'P95 (px)':<12}\n"
        else:
            header = f"{'Model':<25} {'MRE':<12} {'Std':<12} {'Median':<12} {'P90':<12} {'P95':<12} {'Max':<12}\n"
        
        f.write(header)
        f.write("-" * len(header) + "\n")
        
        # Write test set results
        for model_name in ['Ensemble HRNet (Test)', 'Ensemble MLP (Test)']:
            if model_name in results:
                overall = results[model_name]['overall']
                if has_mm_metrics and 'mre_mm' in overall:
                    f.write(f"{model_name:<25} {overall['mre']:<12.3f} {overall.get('mre_mm', 'N/A'):<12.3f} "
                           f"{overall['std']:<12.3f} {overall.get('std_mm', 'N/A'):<12.3f} "
                           f"{overall['median']:<12.3f} {overall['p95']:<12.3f}\n")
                else:
                    f.write(f"{model_name:<25} {overall['mre']:<12.3f} {overall['std']:<12.3f} "
                           f"{overall['median']:<12.3f} {overall['p90']:<12.3f} "
                           f"{overall['p95']:<12.3f} {overall['max']:<12.3f}\n")
        
        # Improvement analysis
        if 'Ensemble HRNet (Test)' in results and 'Ensemble MLP (Test)' in results:
            hrnet_mre = results['Ensemble HRNet (Test)']['overall']['mre']
            mlp_mre = results['Ensemble MLP (Test)']['overall']['mre']
            improvement = (hrnet_mre - mlp_mre) / hrnet_mre * 100
            f.write(f"\nMLP Improvement over HRNetV2: {improvement:+.2f}%\n")
            
            if has_mm_metrics and 'mre_mm' in results['Ensemble MLP (Test)']['overall']:
                hrnet_mre_mm = results['Ensemble HRNet (Test)']['overall'].get('mre_mm', 0)
                mlp_mre_mm = results['Ensemble MLP (Test)']['overall'].get('mre_mm', 0)
                if hrnet_mre_mm > 0:
                    improvement_mm = (hrnet_mre_mm - mlp_mre_mm) / hrnet_mre_mm * 100
                    f.write(f"MLP Improvement in mm: {improvement_mm:+.2f}%\n")
        
        # Individual model results if available
        if args.evaluate_individual:
            f.write("\n\nINDIVIDUAL MODEL PERFORMANCE (TEST SET):\n")
            f.write("-" * 30 + "\n")
            
            individual_models = sorted([k for k in results.keys() if k.startswith('Model') and '(Test)' in k])
            if individual_models:
                f.write(header)
                f.write("-" * len(header) + "\n")
                
                for model_name in individual_models:
                    overall = results[model_name]['overall']
                    if has_mm_metrics and 'mre_mm' in overall:
                        f.write(f"{model_name:<25} {overall['mre']:<12.3f} {overall.get('mre_mm', 'N/A'):<12.3f} "
                               f"{overall['std']:<12.3f} {overall.get('std_mm', 'N/A'):<12.3f} "
                               f"{overall['median']:<12.3f} {overall['p95']:<12.3f}\n")
                    else:
                        f.write(f"{model_name:<25} {overall['mre']:<12.3f} {overall['std']:<12.3f} "
                               f"{overall['median']:<12.3f} {overall['p90']:<12.3f} "
                               f"{overall['p95']:<12.3f} {overall['max']:<12.3f}\n")
        
        # Validation results if available
        if validation_results:
            f.write("\n\nVALIDATION SET PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'Model':<25} {'MRE':<12} {'Std':<12} {'Median':<12} {'P90':<12} {'P95':<12}\n")
            f.write("-" * 80 + "\n")
            
            for model_name, metrics in validation_results.items():
                overall = metrics['overall']
                f.write(f"{model_name:<25} {overall['mre']:<12.3f} {overall['std']:<12.3f} "
                       f"{overall['median']:<12.3f} {overall['p90']:<12.3f} {overall['p95']:<12.3f}\n")
        
        # Define key landmarks and available landmarks (needed for both text and JSON report)
        key_landmarks = ['sella', 'Gonion', 'PNS', 'A_point', 'B_point', 'ANS', 'nasion']
        available_landmarks = [lm for lm in key_landmarks if lm in landmark_names]
        
        # Key landmark performance
        f.write("\n\nKEY LANDMARK PERFORMANCE (TEST SET):\n")
        f.write("-" * 30 + "\n")
        
        for landmark in available_landmarks:
            f.write(f"\n{landmark.upper()}:\n")
            
            # Check if mm metrics are available for this landmark
            has_landmark_mm = any(landmark in metrics['per_landmark'] and 'mre_mm' in metrics['per_landmark'][landmark] 
                                 for metrics in results.values())
            
            if has_landmark_mm:
                f.write(f"{'Model':<25} {'MRE (px)':<12} {'MRE (mm)':<12} {'2mm Acc':<10} {'4mm Acc':<10} {'Count':<8}\n")
            else:
                f.write(f"{'Model':<25} {'MRE':<12} {'Std':<12} {'Median':<12}\n")
            f.write("-" * 75 + "\n")
            
            for model_name in ['Ensemble HRNet (Test)', 'Ensemble MLP (Test)']:
                if model_name in results and landmark in results[model_name]['per_landmark']:
                    lm_metrics = results[model_name]['per_landmark'][landmark]
                    if has_landmark_mm and 'mre_mm' in lm_metrics:
                        acc_2mm = lm_metrics.get('accuracy_2mm', 0) * 100
                        acc_4mm = lm_metrics.get('accuracy_4mm', 0) * 100
                        count = lm_metrics.get('accuracy_total_count', lm_metrics.get('count', 0))
                        f.write(f"{model_name:<25} {lm_metrics['mre']:<12.3f} {lm_metrics.get('mre_mm', 'N/A'):<12.3f} "
                               f"{acc_2mm:<10.1f}% {acc_4mm:<10.1f}% {count:<8}\n")
                    else:
                        f.write(f"{model_name:<25} {lm_metrics['mre']:<12.3f} {lm_metrics['std']:<12.3f} "
                               f"{lm_metrics['median']:<12.3f}\n")
        
        # All landmarks performance
        f.write("\n\nALL LANDMARKS PERFORMANCE (TEST SET):\n")
        f.write("-" * 30 + "\n")
        
        # Overall accuracy summary
        if has_mm_metrics:
            f.write("Overall accuracy metrics based on clinical error thresholds:\n\n")
            
            for model_name in ['Ensemble HRNet (Test)', 'Ensemble MLP (Test)']:
                if model_name in results:
                    overall = results[model_name]['overall']
                    if 'accuracy_2mm' in overall and 'accuracy_4mm' in overall:
                        acc_2mm = overall['accuracy_2mm'] * 100
                        acc_4mm = overall['accuracy_4mm'] * 100
                        total_count = overall.get('accuracy_total_count', 0)
                        f.write(f"{model_name}:\n")
                        f.write(f"  2mm Accuracy: {acc_2mm:.1f}% ({overall.get('accuracy_2mm_count', 0)}/{total_count})\n")
                        f.write(f"  4mm Accuracy: {acc_4mm:.1f}% ({overall.get('accuracy_4mm_count', 0)}/{total_count})\n")
                        f.write(f"  Calibrated Patients: {overall.get('calibrated_patients', 0)}\n")
                        f.write("\n")
        
        # Detailed landmark performance table
        if has_mm_metrics:
            f.write(f"{'Landmark':<20} {'Model':<25} {'MRE (mm)':<10} {'2mm Acc':<10} {'4mm Acc':<10} {'Count':<8}\n")
        else:
            f.write(f"{'Landmark':<20} {'Model':<25} {'MRE (px)':<10} {'Std (px)':<10} {'Median':<10} {'Count':<8}\n")
        f.write("-" * 95 + "\n")
        
        for landmark in landmark_names:
            landmark_written = False
            for model_name in ['Ensemble HRNet (Test)', 'Ensemble MLP (Test)']:
                if model_name in results and landmark in results[model_name]['per_landmark']:
                    lm_metrics = results[model_name]['per_landmark'][landmark]
                    landmark_display = landmark if not landmark_written else ""
                    landmark_written = True
                    
                    if has_mm_metrics and 'mre_mm' in lm_metrics:
                        acc_2mm = lm_metrics.get('accuracy_2mm', 0) * 100
                        acc_4mm = lm_metrics.get('accuracy_4mm', 0) * 100
                        count = lm_metrics.get('accuracy_total_count', lm_metrics.get('count', 0))
                        f.write(f"{landmark_display:<20} {model_name:<25} {lm_metrics.get('mre_mm', 'N/A'):<10.3f} "
                               f"{acc_2mm:<10.1f}% {acc_4mm:<10.1f}% {count:<8}\n")
                    else:
                        f.write(f"{landmark_display:<20} {model_name:<25} {lm_metrics['mre']:<10.3f} "
                               f"{lm_metrics['std']:<10.3f} {lm_metrics['median']:<10.3f} "
                               f"{lm_metrics['count']:<8}\n")
            if landmark_written:
                f.write("\n")
        
        # All angles performance (need to read from angle CSV files if they exist)
        f.write("\n\nALL ANGLES PERFORMANCE (TEST SET):\n")
        f.write("-" * 30 + "\n")
        
        # Try to load angle data from the ensemble angle predictions file
        angle_csv_path = os.path.join(output_dir, "ensemble_angle_predictions.csv")
        if os.path.exists(angle_csv_path):
            try:
                angle_df = pd.read_csv(angle_csv_path)
                angle_names = ['SNA', 'SNB', 'ANB', 'u1', 'l1', 'sn_ans_pns', 'sn_mn_go', 'nasolabial_angle']
                
                f.write("Accuracy metrics based on 2° and 4° acceptable error thresholds:\n\n")
                
                for model_name in ['ensemble_hrnetv2', 'ensemble_mlp']:
                    model_display = 'Ensemble HRNetV2' if model_name == 'ensemble_hrnetv2' else 'Ensemble MLP'
                    f.write(f"{model_display}:\n")
                    
                    total_angle_predictions = 0
                    accurate_2deg_predictions = 0
                    accurate_4deg_predictions = 0
                    
                    for angle_name in angle_names:
                        error_col = f'{model_name}_{angle_name}_error'
                        if error_col in angle_df.columns:
                            errors = angle_df[error_col].dropna()
                            if len(errors) > 0:
                                accurate_2deg_count = (errors <= 2.0).sum()
                                accurate_4deg_count = (errors <= 4.0).sum()
                                total_count = len(errors)
                                total_angle_predictions += total_count
                                accurate_2deg_predictions += accurate_2deg_count
                                accurate_4deg_predictions += accurate_4deg_count
                    
                    if total_angle_predictions > 0:
                        angle_accuracy_2deg = accurate_2deg_predictions / total_angle_predictions
                        angle_accuracy_4deg = accurate_4deg_predictions / total_angle_predictions
                        f.write(f"  Overall 2° Accuracy: {angle_accuracy_2deg:.1%} ({accurate_2deg_predictions}/{total_angle_predictions})\n")
                        f.write(f"  Overall 4° Accuracy: {angle_accuracy_4deg:.1%} ({accurate_4deg_predictions}/{total_angle_predictions})\n")
                    f.write("\n")
                
                # Detailed angle performance table
                f.write(f"{'Angle':<20} {'Model':<25} {'Mean Error (°)':<15} {'2° Accuracy':<12} {'4° Accuracy':<12} {'Count':<8}\n")
                f.write("-" * 110 + "\n")
                
                for angle_name in angle_names:
                    angle_written = False
                    for model_name in ['ensemble_hrnetv2', 'ensemble_mlp']:
                        model_display = 'Ensemble HRNetV2' if model_name == 'ensemble_hrnetv2' else 'Ensemble MLP'
                        error_col = f'{model_name}_{angle_name}_error'
                        
                        if error_col in angle_df.columns:
                            errors = angle_df[error_col].dropna()
                            if len(errors) > 0:
                                angle_display = angle_name if not angle_written else ""
                                angle_written = True
                                
                                mean_error = errors.mean()
                                accurate_2deg_count = (errors <= 2.0).sum()
                                accurate_4deg_count = (errors <= 4.0).sum()
                                total_count = len(errors)
                                accuracy_2deg = accurate_2deg_count / total_count if total_count > 0 else 0
                                accuracy_4deg = accurate_4deg_count / total_count if total_count > 0 else 0
                                
                                f.write(f"{angle_display:<20} {model_display:<25} {mean_error:<15.2f} "
                                       f"{accuracy_2deg:<12.1%} {accuracy_4deg:<12.1%} {total_count:<8}\n")
                    if angle_written:
                        f.write("\n")
                        
            except Exception as e:
                f.write(f"Could not load angle data: {e}\n")
        else:
            f.write("Angle data not available (ensemble_angle_predictions.csv not found)\n")
        
        # Classification results if available
        if classification_results:
            f.write("\n\nPATIENT CLASSIFICATION RESULTS (ANB-based):\n")
            f.write("-" * 30 + "\n")
            
            for model_type in ['ensemble_hrnetv2', 'ensemble_mlp']:
                if model_type in classification_results:
                    metrics = classification_results[model_type]
                    model_name = 'Ensemble HRNetV2' if model_type == 'ensemble_hrnetv2' else 'Ensemble MLP'
                    
                    f.write(f"\n{model_name}:\n")
                    f.write(f"  Accuracy: {metrics['accuracy']:.3f}\n")
                    f.write(f"  Precision (macro): {metrics['precision']:.3f}\n")
                    f.write(f"  Recall (macro): {metrics['recall']:.3f}\n")
                    f.write(f"  F1-Score (macro): {metrics['f1_score']:.3f}\n")
                    f.write(f"  Valid samples: {metrics['n_samples']}\n")
                    
                    # Per-class metrics
                    if 'per_class' in metrics and metrics['per_class']:
                        f.write("\n  Per-Class Performance:\n")
                        for class_name, class_metrics in metrics['per_class'].items():
                            f.write(f"    {class_name}: "
                                   f"Precision={class_metrics['precision']:.3f}, "
                                   f"Recall={class_metrics['recall']:.3f}, "
                                   f"F1={class_metrics['f1']:.3f}, "
                                   f"Support={class_metrics['support']}\n")
        
        # Summary statistics
        f.write("\n\nSUMMARY:\n")
        f.write("-" * 30 + "\n")
        
        if 'Ensemble MLP (Test)' in results:
            ensemble_mre = results['Ensemble MLP (Test)']['overall']['mre']
            ensemble_mre_600 = ensemble_mre * SCALE_FACTOR
            f.write(f"Ensemble MLP Mean Radial Error:\n")
            f.write(f"  - {ensemble_mre:.3f} pixels (224x224 space)\n")
            f.write(f"  - {ensemble_mre_600:.3f} pixels (600x600 space)\n")
            
            if 'mre_mm' in results['Ensemble MLP (Test)']['overall']:
                ensemble_mre_mm = results['Ensemble MLP (Test)']['overall']['mre_mm']
                calibrated_patients = results['Ensemble MLP (Test)']['overall'].get('calibrated_patients', 0)
                f.write(f"  - {ensemble_mre_mm:.3f} mm (from {calibrated_patients} patients with calibration)\n")
        
        # Add accuracy summary
        f.write(f"\nAccuracy Metrics Summary:\n")
        
        # Landmark accuracy summary
        if has_mm_metrics:
            for model_name in ['Ensemble HRNet (Test)', 'Ensemble MLP (Test)']:
                if model_name in results:
                    overall = results[model_name]['overall']
                    if 'accuracy_2mm' in overall and 'accuracy_4mm' in overall:
                        model_short = 'HRNetV2' if 'HRNet' in model_name else 'MLP'
                        acc_2mm = overall['accuracy_2mm'] * 100
                        acc_4mm = overall['accuracy_4mm'] * 100
                        f.write(f"  - {model_short} Landmark 2mm Accuracy: {acc_2mm:.1f}%\n")
                        f.write(f"  - {model_short} Landmark 4mm Accuracy: {acc_4mm:.1f}%\n")
        
        # Angle accuracy summary
        angle_csv_path = os.path.join(output_dir, "ensemble_angle_predictions.csv")
        if os.path.exists(angle_csv_path):
            try:
                angle_df = pd.read_csv(angle_csv_path)
                angle_names = ['SNA', 'SNB', 'ANB', 'u1', 'l1', 'sn_ans_pns', 'sn_mn_go', 'nasolabial_angle']
                
                for model_name in ['ensemble_hrnetv2', 'ensemble_mlp']:
                    total_angle_predictions = 0
                    accurate_2deg_predictions = 0
                    accurate_4deg_predictions = 0
                    
                    for angle_name in angle_names:
                        error_col = f'{model_name}_{angle_name}_error'
                        if error_col in angle_df.columns:
                            errors = angle_df[error_col].dropna()
                            if len(errors) > 0:
                                accurate_2deg_count = (errors <= 2.0).sum()
                                accurate_4deg_count = (errors <= 4.0).sum()
                                total_count = len(errors)
                                total_angle_predictions += total_count
                                accurate_2deg_predictions += accurate_2deg_count
                                accurate_4deg_predictions += accurate_4deg_count
                    
                    if total_angle_predictions > 0:
                        angle_accuracy_2deg = accurate_2deg_predictions / total_angle_predictions
                        angle_accuracy_4deg = accurate_4deg_predictions / total_angle_predictions
                        model_short = 'HRNetV2' if model_name == 'ensemble_hrnetv2' else 'MLP'
                        f.write(f"  - {model_short} Angle 2° Accuracy: {angle_accuracy_2deg:.1%}\n")
                        f.write(f"  - {model_short} Angle 4° Accuracy: {angle_accuracy_4deg:.1%}\n")
            except:
                pass
        
        # File locations
        f.write("\n\nOUTPUT FILES:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Results saved to: {output_dir}\n")
        f.write("\nKey files:\n")
        f.write("  - overall_results_report.txt (this file)\n")
        f.write("  - ensemble_mlp_predictions_detailed.csv\n")
        f.write("  - ensemble_hrnetv2_predictions_detailed.csv\n")
        f.write("  - ensemble_angle_predictions.csv\n")
        f.write("  - all_models_angle_predictions.csv\n")
        f.write("  - angle_error_analysis.png\n")
        f.write("  - classification_analysis.png\n")
        f.write("  - patient_visualizations/\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"\n📄 Overall results report saved to: {os.path.basename(report_path)}")
    
    # Helper function to convert NumPy arrays to lists for JSON serialization
    def convert_numpy_to_list(obj):
        """Recursively convert NumPy arrays to Python lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_to_list(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_to_list(item) for item in obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    # Also save as JSON for programmatic access
    json_report_path = os.path.join(output_dir, "overall_results.json")
    json_report = {
        'configuration': {
            'timestamp': pd.Timestamp.now().isoformat(),
            'base_work_dir': args.base_work_dir,
            'n_models': args.n_models,
            'n_models_evaluated': n_models_evaluated,
            'evaluation_mode': f'epoch_{args.epoch}' if args.epoch else 'best_latest',
            'epoch': args.epoch,
            'test_samples': n_test_samples,
            'evaluate_individual': args.evaluate_individual,
            'evaluate_on_validation': args.evaluate_on_validation,
            'image_scaling': {
                'input_size': MODEL_INPUT_SIZE,
                'output_size': ORIGINAL_IMAGE_SIZE,
                'scale_factor': SCALE_FACTOR
            },
            'calibration_available': ruler_data is not None
        },
        'test_results': {},
        'validation_results': {},
        'classification_results': convert_numpy_to_list(classification_results)
    }
    
    # Add test results
    for model_name, metrics in results.items():
        json_report['test_results'][model_name] = convert_numpy_to_list({
            'overall': metrics['overall'],
            'key_landmarks': {lm: metrics['per_landmark'].get(lm, {}) 
                            for lm in available_landmarks if lm in metrics['per_landmark']},
            'all_landmarks': metrics['per_landmark']
        })
    
    # Add validation results
    for model_name, metrics in validation_results.items():
        json_report['validation_results'][model_name] = convert_numpy_to_list({
            'overall': metrics['overall']
        })
    
    # Add accuracy metrics and angle data to JSON report
    json_report['accuracy_metrics'] = {}
    
    # Add accuracy metrics for landmarks (2mm and 4mm)
    if results and ruler_data:
        json_report['accuracy_metrics']['landmark_accuracy'] = {}
        for model_name in ['Ensemble HRNet (Test)', 'Ensemble MLP (Test)']:
            if model_name in results:
                overall = results[model_name]['overall']
                if 'accuracy_2mm' in overall and 'accuracy_4mm' in overall:
                    json_report['accuracy_metrics']['landmark_accuracy'][model_name] = {
                        'accuracy_2mm': float(overall['accuracy_2mm']),
                        'accuracy_4mm': float(overall['accuracy_4mm']),
                        'accurate_2mm_count': int(overall.get('accuracy_2mm_count', 0)),
                        'accurate_4mm_count': int(overall.get('accuracy_4mm_count', 0)),
                        'total_predictions': int(overall.get('accuracy_total_count', 0))
                    }
    
    # Add angle data and 2-degree accuracy
    angle_csv_path = os.path.join(output_dir, "ensemble_angle_predictions.csv")
    if os.path.exists(angle_csv_path):
        try:
            angle_df = pd.read_csv(angle_csv_path)
            angle_names = ['SNA', 'SNB', 'ANB', 'u1', 'l1', 'sn_ans_pns', 'sn_mn_go', 'nasolabial_angle']
            
            json_report['angle_results'] = {}
            json_report['accuracy_metrics']['angle_accuracy'] = {}
            
            for model_name in ['ensemble_hrnetv2', 'ensemble_mlp']:
                model_display = 'Ensemble HRNetV2' if model_name == 'ensemble_hrnetv2' else 'Ensemble MLP'
                json_report['angle_results'][model_display] = {}
                
                total_angle_predictions = 0
                accurate_2deg_predictions = 0
                accurate_4deg_predictions = 0
                
                for angle_name in angle_names:
                    error_col = f'{model_name}_{angle_name}_error'
                    if error_col in angle_df.columns:
                        errors = angle_df[error_col].dropna()
                        if len(errors) > 0:
                            mean_error = errors.mean()
                            std_error = errors.std()
                            accurate_2deg_count = (errors <= 2.0).sum()
                            accurate_4deg_count = (errors <= 4.0).sum()
                            total_count = len(errors)
                            accuracy_2deg = accurate_2deg_count / total_count if total_count > 0 else 0
                            accuracy_4deg = accurate_4deg_count / total_count if total_count > 0 else 0
                            
                            json_report['angle_results'][model_display][angle_name] = {
                                'mean_error': float(mean_error),
                                'std_error': float(std_error),
                                'accuracy_2deg': float(accuracy_2deg),
                                'accuracy_4deg': float(accuracy_4deg),
                                'accurate_2deg_count': int(accurate_2deg_count),
                                'accurate_4deg_count': int(accurate_4deg_count),
                                'total_count': int(total_count)
                            }
                            
                            total_angle_predictions += total_count
                            accurate_2deg_predictions += accurate_2deg_count
                            accurate_4deg_predictions += accurate_4deg_count
                
                if total_angle_predictions > 0:
                    overall_angle_accuracy_2deg = accurate_2deg_predictions / total_angle_predictions
                    overall_angle_accuracy_4deg = accurate_4deg_predictions / total_angle_predictions
                    json_report['accuracy_metrics']['angle_accuracy'][model_display] = {
                        'accuracy_2deg': float(overall_angle_accuracy_2deg),
                        'accuracy_4deg': float(overall_angle_accuracy_4deg),
                        'accurate_2deg_count': int(accurate_2deg_predictions),
                        'accurate_4deg_count': int(accurate_4deg_predictions),
                        'total_predictions': int(total_angle_predictions)
                    }
                    
        except Exception as e:
            json_report['angle_results'] = {'error': f"Could not load angle data: {e}"}
    
    with open(json_report_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    
    print(f"📄 JSON results saved to: {os.path.basename(json_report_path)}")

def create_classification_visualization(classification_results: Dict[str, Dict], output_dir: str):
    """Create visualization of patient classification results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract metrics
    hrnet_metrics = classification_results['ensemble_hrnetv2']
    mlp_metrics = classification_results['ensemble_mlp']
    
    # Plot 1: Confusion Matrices
    ax = axes[0, 0]
    if hrnet_metrics['confusion_matrix'] is not None:
        im = ax.imshow(hrnet_metrics['confusion_matrix'], cmap='Blues', interpolation='nearest')
        
        # Add text annotations
        for i in range(len(hrnet_metrics['class_names'])):
            for j in range(len(hrnet_metrics['class_names'])):
                text = ax.text(j, i, str(hrnet_metrics['confusion_matrix'][i, j]),
                             ha="center", va="center", color="black")
        
        ax.set_xticks(range(len(hrnet_metrics['class_names'])))
        ax.set_yticks(range(len(hrnet_metrics['class_names'])))
        ax.set_xticklabels(hrnet_metrics['class_names'])
        ax.set_yticklabels(hrnet_metrics['class_names'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Ensemble HRNetV2 Confusion Matrix')
        plt.colorbar(im, ax=ax)
    
    ax = axes[0, 1]
    if mlp_metrics['confusion_matrix'] is not None:
        im = ax.imshow(mlp_metrics['confusion_matrix'], cmap='Reds', interpolation='nearest')
        
        # Add text annotations
        for i in range(len(mlp_metrics['class_names'])):
            for j in range(len(mlp_metrics['class_names'])):
                text = ax.text(j, i, str(mlp_metrics['confusion_matrix'][i, j]),
                             ha="center", va="center", color="black")
        
        ax.set_xticks(range(len(mlp_metrics['class_names'])))
        ax.set_yticks(range(len(mlp_metrics['class_names'])))
        ax.set_xticklabels(mlp_metrics['class_names'])
        ax.set_yticklabels(mlp_metrics['class_names'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Ensemble MLP Confusion Matrix')
        plt.colorbar(im, ax=ax)
    
    # Plot 2: Overall metrics comparison
    ax = axes[1, 0]
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    hrnet_values = [hrnet_metrics['accuracy'], hrnet_metrics['precision'], 
                    hrnet_metrics['recall'], hrnet_metrics['f1_score']]
    mlp_values = [mlp_metrics['accuracy'], mlp_metrics['precision'], 
                  mlp_metrics['recall'], mlp_metrics['f1_score']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, hrnet_values, width, label='Ensemble HRNetV2', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, mlp_values, width, label='Ensemble MLP', color='red', alpha=0.7)
    
    ax.set_ylabel('Score')
    ax.set_title('Classification Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=8)
    
    # Plot 3: Per-class performance
    ax = axes[1, 1]
    
    # Get per-class metrics
    classes = hrnet_metrics['class_names']
    per_class_data = []
    
    for class_name in classes:
        hrnet_class = hrnet_metrics['per_class'].get(class_name, {})
        mlp_class = mlp_metrics['per_class'].get(class_name, {})
        
        per_class_data.append({
            'class': class_name,
            'hrnet_f1': hrnet_class.get('f1', 0),
            'mlp_f1': mlp_class.get('f1', 0),
            'support': hrnet_class.get('support', 0)
        })
    
    if per_class_data:
        class_names = [d['class'] for d in per_class_data]
        hrnet_f1s = [d['hrnet_f1'] for d in per_class_data]
        mlp_f1s = [d['mlp_f1'] for d in per_class_data]
        supports = [d['support'] for d in per_class_data]
        
        x = np.arange(len(class_names))
        bars1 = ax.bar(x - width/2, hrnet_f1s, width, label='Ensemble HRNetV2', color='blue', alpha=0.7)
        bars2 = ax.bar(x + width/2, mlp_f1s, width, label='Ensemble MLP', color='red', alpha=0.7)
        
        ax.set_ylabel('F1-Score')
        ax.set_title('Per-Class F1-Score Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        # Add sample counts
        for i, (x_pos, support) in enumerate(zip(x, supports)):
            ax.text(x_pos, -0.05, f'n={support}', ha='center', va='top', fontsize=8)
    
    plt.suptitle('Patient Classification Analysis (ANB-based)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'classification_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Classification visualization saved to: {os.path.basename(output_path)}")

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
    parser.add_argument(
        '--evaluate_on_validation',
        action='store_true',
        help='Also evaluate each model on its own validation set'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=None,
        help='Specific epoch number to evaluate (e.g., --epoch 20 will use epoch_20.pth checkpoints from all models)'
    )
    args = parser.parse_args()
    
    print("="*80)
    print("ENSEMBLE CONCURRENT JOINT MLP EVALUATION")
    print("="*80)
    print(f"📏 Image scaling: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE} → {ORIGINAL_IMAGE_SIZE}x{ORIGINAL_IMAGE_SIZE} (scale factor: {SCALE_FACTOR:.4f})")
    
    if args.epoch is not None:
        print(f"🎯 Evaluating at specific epoch: {args.epoch}")
    else:
        print(f"🎯 Using best/latest checkpoints from each model")
    
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
    
    # Load ruler calibration data
    ruler_data_path = "data/patient_ruler_data.json"
    ruler_data = load_ruler_calibration_data(ruler_data_path)
    
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
    all_patient_ids = None
    results = {}
    validation_results = {}
    
    for i, model_dir in enumerate(model_dirs, 1):
        components = load_model_components(model_dir, device, config_path, args.epoch)
        
        if components is None:
            print(f"   ❌ Skipping model {i} due to loading errors")
            continue
        
        hrnet_model, mlp_joint, scaler_input, scaler_target, model_type, checkpoint_name = components
        all_model_components.append((hrnet_model, mlp_joint, scaler_input, scaler_target, model_type, checkpoint_name))
        
        # Evaluate this model on test set
        hrnet_preds, mlp_preds, gt_coords, patient_ids = evaluate_single_model(
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
        
        if all_patient_ids is None:
            all_patient_ids = patient_ids
        
        # Compute metrics for individual model if requested
        if args.evaluate_individual:
            if ruler_data:
                hrnet_overall, hrnet_per_landmark = compute_metrics_with_mm(hrnet_preds, gt_coords, landmark_names, patient_ids, ruler_data)
                mlp_overall, mlp_per_landmark = compute_metrics_with_mm(mlp_preds, gt_coords, landmark_names, patient_ids, ruler_data)
            else:
                hrnet_overall, hrnet_per_landmark = compute_metrics(hrnet_preds, gt_coords, landmark_names)
                mlp_overall, mlp_per_landmark = compute_metrics(mlp_preds, gt_coords, landmark_names)
            
            results[f'Model {i} HRNet (Test)'] = {'overall': hrnet_overall, 'per_landmark': hrnet_per_landmark}
            results[f'Model {i} MLP (Test)'] = {'overall': mlp_overall, 'per_landmark': mlp_per_landmark}
        
        # Evaluate on validation set if requested
        if args.evaluate_on_validation:
            print(f"   🔄 Evaluating Model {i} on its validation set...")
            
            # Load validation data for this model
            val_ann_file = os.path.join(model_dir, f'temp_val_ann_split_{i}.json')
            
            if os.path.exists(val_ann_file):
                try:
                    val_df = pd.read_json(val_ann_file)
                    print(f"      ✓ Loaded validation set: {len(val_df)} samples")
                    
                    # Evaluate on validation set
                    val_hrnet_preds, val_mlp_preds, val_gt_coords, val_patient_ids = evaluate_single_model(
                        hrnet_model, mlp_joint, scaler_input, scaler_target,
                        val_df, landmark_names, landmark_cols, device
                    )
                    
                    if val_hrnet_preds is not None:
                        print(f"      ✓ Model {i} validation evaluation: {len(val_hrnet_preds)} samples")
                        
                        # Compute validation metrics
                        val_hrnet_overall, val_hrnet_per_landmark = compute_metrics(val_hrnet_preds, val_gt_coords, landmark_names)
                        val_mlp_overall, val_mlp_per_landmark = compute_metrics(val_mlp_preds, val_gt_coords, landmark_names)
                        
                        validation_results[f'Model {i} HRNet (Val)'] = {'overall': val_hrnet_overall, 'per_landmark': val_hrnet_per_landmark}
                        validation_results[f'Model {i} MLP (Val)'] = {'overall': val_mlp_overall, 'per_landmark': val_mlp_per_landmark}
                    else:
                        print(f"      ❌ No valid validation predictions from model {i}")
                        
                except Exception as e:
                    print(f"      ❌ Failed to load validation data for model {i}: {e}")
            else:
                print(f"      ⚠️  Validation file not found: {val_ann_file}")
    
    if not all_hrnet_preds:
        print("ERROR: No models successfully evaluated")
        return
    
    # Create ensemble predictions
    ensemble_hrnet, ensemble_mlp = create_ensemble_predictions(all_hrnet_preds, all_mlp_preds)
    
    # Evaluate ensemble
    print(f"\n🔄 Computing ensemble metrics...")
    if ruler_data:
        ensemble_hrnet_overall, ensemble_hrnet_per_landmark = compute_metrics_with_mm(ensemble_hrnet, all_gt, landmark_names, all_patient_ids, ruler_data)
        ensemble_mlp_overall, ensemble_mlp_per_landmark = compute_metrics_with_mm(ensemble_mlp, all_gt, landmark_names, all_patient_ids, ruler_data)
    else:
        ensemble_hrnet_overall, ensemble_hrnet_per_landmark = compute_metrics(ensemble_hrnet, all_gt, landmark_names)
        ensemble_mlp_overall, ensemble_mlp_per_landmark = compute_metrics(ensemble_mlp, all_gt, landmark_names)
    
    results['Ensemble HRNet (Test)'] = {'overall': ensemble_hrnet_overall, 'per_landmark': ensemble_hrnet_per_landmark}
    results['Ensemble MLP (Test)'] = {'overall': ensemble_mlp_overall, 'per_landmark': ensemble_mlp_per_landmark}
    
    # Print results
    print_results_table(results, landmark_names)
    
    # Print validation results if available
    if validation_results:
        print(f"\n{'='*100}")
        print(f"📊 VALIDATION SET RESULTS")
        print(f"{'='*100}")
        print_results_table(validation_results, landmark_names)
    
    # Save results
    output_dir = os.path.join(args.base_work_dir, "ensemble_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save ensemble predictions to CSV with patient details
    save_ensemble_predictions_to_csv(ensemble_hrnet, ensemble_mlp, all_gt, all_patient_ids, landmark_names, output_dir, ruler_data)
    
    # Save detailed comparison (test results)
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
    comparison_df.to_csv(os.path.join(output_dir, "ensemble_comparison_test.csv"), index=False)
    
    # Save validation comparison if available
    if validation_results:
        val_comparison_data = []
        for result_name, metrics in validation_results.items():
            val_comparison_data.append({
                'model': result_name,
                'mre': metrics['overall']['mre'],
                'std': metrics['overall']['std'],
                'median': metrics['overall']['median'],
                'p90': metrics['overall']['p90'],
                'p95': metrics['overall']['p95'],
                'samples': metrics['overall']['count']
            })
        
        val_comparison_df = pd.DataFrame(val_comparison_data)
        val_comparison_df.to_csv(os.path.join(output_dir, "ensemble_comparison_validation.csv"), index=False)
    
    # Save per-landmark results for key landmarks (test)
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
    landmark_df.to_csv(os.path.join(output_dir, "key_landmarks_comparison_test.csv"), index=False)
    
    # Save validation landmark results if available
    if validation_results:
        val_landmark_data = []
        
        for landmark in key_landmarks:
            if landmark in landmark_names:
                for result_name, metrics in validation_results.items():
                    if landmark in metrics['per_landmark']:
                        lm_metrics = metrics['per_landmark'][landmark]
                        val_landmark_data.append({
                            'model': result_name,
                            'landmark': landmark,
                            'mre': lm_metrics['mre'],
                            'std': lm_metrics['std'],
                            'median': lm_metrics['median'],
                            'count': lm_metrics['count']
                        })
        
        val_landmark_df = pd.DataFrame(val_landmark_data)
        val_landmark_df.to_csv(os.path.join(output_dir, "key_landmarks_comparison_validation.csv"), index=False)
    
    # Save individual model predictions (using already computed predictions)
    print(f"\n💾 Saving individual model predictions...")
    for i in range(len(all_hrnet_preds)):
        save_individual_model_predictions(i+1, all_hrnet_preds[i], all_mlp_preds[i], 
                                        all_gt, all_patient_ids, landmark_names, output_dir, ruler_data)
    
    # Save all models combined
    save_all_models_combined(all_hrnet_preds, all_mlp_preds, ensemble_hrnet, ensemble_mlp, all_gt, all_patient_ids, landmark_names, output_dir, ruler_data)
    
        # Save angle predictions and get classification results
    classification_results = save_angle_predictions_to_csv(ensemble_hrnet, ensemble_mlp, all_hrnet_preds, all_mlp_preds, all_gt, all_patient_ids, landmark_names, output_dir, ruler_data)
    
    # Create patient visualizations
    create_patient_visualizations(ensemble_hrnet, ensemble_mlp, all_hrnet_preds, all_mlp_preds, all_gt, all_patient_ids, test_df, landmark_names, output_dir, ruler_data)
    
    # Save overall results report
    save_overall_results_report(
        results=results,
        validation_results=validation_results,
        classification_results=classification_results,
        landmark_names=landmark_names,
        args=args,
        output_dir=output_dir,
        n_test_samples=len(test_df),
        n_models_evaluated=len(all_hrnet_preds),
        ruler_data=ruler_data
    )
    
    print(f"\n💾 Results saved to: {output_dir}")
    print(f"\n📊 Summary Files:")
    print(f"   - Overall results report: overall_results_report.txt")
    print(f"   - Overall results JSON: overall_results.json")
    print(f"   - Test set comparison: ensemble_comparison_test.csv")
    print(f"   - Test set landmarks: key_landmarks_comparison_test.csv")
    if validation_results:
        print(f"   - Validation comparison: ensemble_comparison_validation.csv")
        print(f"   - Validation landmarks: key_landmarks_comparison_validation.csv")
    
    print(f"\n📋 Detailed Prediction Files:")
    print(f"   - Ensemble MLP predictions: ensemble_mlp_predictions_detailed.csv")
    print(f"   - Ensemble HRNetV2 predictions: ensemble_hrnetv2_predictions_detailed.csv")
    
    print(f"\n📁 Individual Model Files:")
    for i in range(len(all_hrnet_preds)):
        print(f"   - Model {i+1} MLP: model{i+1}_mlp_predictions_detailed.csv")
        print(f"   - Model {i+1} HRNetV2: model{i+1}_hrnetv2_predictions_detailed.csv")
    
    print(f"\n📊 Combined Analysis Files:")
    print(f"   - All models MLP combined: all_models_mlp_predictions_combined.csv")
    print(f"   - All models HRNetV2 combined: all_models_hrnetv2_predictions_combined.csv")
    
    print(f"\n🎨 Patient Visualizations:")
    print(f"   - Best 5 patients: best_1_patient_*.png ... best_5_patient_*.png")
    print(f"   - Worst 3 patients: worst_1_patient_*.png ... worst_3_patient_*.png")
    print(f"   - Performance summary: performance_summary.png")
    print(f"   Location: {os.path.join('ensemble_evaluation', 'patient_visualizations')}")
    
    print(f"\n📄 Patient JSON Data:")
    print(f"   - Individual best patients: best_1_patient_*.json ... best_5_patient_*.json")
    print(f"   - Individual worst patients: worst_1_patient_*.json ... worst_3_patient_*.json")
    print(f"   - Combined summary: best_worst_patients_summary.json")
    print(f"   - Structure: ID, classification, landmarks (600x600), ground truth, angles, soft tissue, distances")
    print(f"   - Coordinates: Scaled to original 600x600 image space")
    print(f"   - Distances: Provided in pixels (224x224, 600x600) and mm (when calibration available)")
    print(f"   Location: {os.path.join('ensemble_evaluation', 'patient_visualizations')}")
    
    print(f"\n📐 Cephalometric Angle Files:")
    print(f"   - Ensemble angles & soft tissue: ensemble_angle_predictions.csv")
    print(f"   - All models angles & soft tissue: all_models_angle_predictions.csv")
    print(f"   - Angle error analysis: angle_error_analysis.png")
    print(f"   - Classification analysis: classification_analysis.png")
    print(f"   Note: Files include ANB angle, soft tissue measurements, and patient classifications")
    print(f"   Note: Coordinates are scaled to original 600x600 image space")
    print(f"   Note: Errors are provided in pixels (224x224 and 600x600) and mm (when calibration available)")
    if ruler_data:
        print(f"   Note: Millimeter calibration applied using {RULER_LENGTH_MM}mm ruler measurements")
    
    print(f"\n📈 Enhanced Overall Results Report:")
    print(f"   - Comprehensive text report: overall_results_report.txt")
    print(f"   - Machine-readable JSON: overall_results.json")
    print(f"   - ALL landmarks performance (19 landmarks)")
    print(f"   - ALL angles analysis (8 measurements)")
    print(f"   - Accuracy metrics: 2mm/4mm threshold for landmarks, 2°/4° threshold for angles")
    print(f"   - Clinical acceptability assessment")
    print(f"   - Complete evaluation configuration and metadata")
    
    # Quick summary
    ensemble_mre = ensemble_mlp_overall['mre']
    ensemble_mre_600 = ensemble_mre * SCALE_FACTOR
    print(f"\n🎉 Ensemble Evaluation Summary:")
    if args.epoch is not None:
        print(f"📊 {len(all_hrnet_preds)} models successfully evaluated at epoch {args.epoch}")
    else:
        print(f"📊 {len(all_hrnet_preds)} models successfully evaluated (using best/latest checkpoints)")
    print(f"🎯 Ensemble MLP MRE: {ensemble_mre:.3f} pixels (224x224) / {ensemble_mre_600:.3f} pixels (600x600)")
    
    # Show mm errors if available in metrics
    if 'mre_mm' in ensemble_mlp_overall:
        ensemble_mre_mm = ensemble_mlp_overall['mre_mm']
        calibrated_patients = ensemble_mlp_overall.get('calibrated_patients', 0)
        print(f"🎯 Ensemble MLP MRE: {ensemble_mre_mm:.3f} mm (from {calibrated_patients} patients with calibration)")
        
        if 'mre_mm' in ensemble_hrnet_overall:
            hrnet_mre_mm = ensemble_hrnet_overall['mre_mm']
            improvement_mm = (hrnet_mre_mm - ensemble_mre_mm) / hrnet_mre_mm * 100 if hrnet_mre_mm > 0 else 0
            print(f"🎯 Ensemble HRNetV2 MRE: {hrnet_mre_mm:.3f} mm")
            print(f"📈 MLP improvement over HRNetV2: {improvement_mm:+.1f}% (in mm)")
    
    # Add accuracy metrics to console summary
    print(f"\n🎯 Accuracy Metrics:")
    
    # Calculate and show landmark accuracy
    if 'mre_mm' in ensemble_mlp_overall:
        for model_name in ['Ensemble HRNet (Test)', 'Ensemble MLP (Test)']:
            if model_name in results:
                overall = results[model_name]['overall']
                if 'accuracy_2mm' in overall and 'accuracy_4mm' in overall:
                    model_short = 'HRNetV2' if 'HRNet' in model_name else 'MLP'
                    acc_2mm = overall['accuracy_2mm'] * 100
                    acc_4mm = overall['accuracy_4mm'] * 100
                    print(f"   📍 {model_short} Landmark 2mm Accuracy: {acc_2mm:.1f}%")
                    print(f"   📍 {model_short} Landmark 4mm Accuracy: {acc_4mm:.1f}%")
    
    # Calculate and show angle accuracy
    angle_csv_path = os.path.join(output_dir, "ensemble_angle_predictions.csv")
    if os.path.exists(angle_csv_path):
        try:
            angle_df = pd.read_csv(angle_csv_path)
            angle_names = ['SNA', 'SNB', 'ANB', 'u1', 'l1', 'sn_ans_pns', 'sn_mn_go', 'nasolabial_angle']
            
            for model_name in ['ensemble_hrnetv2', 'ensemble_mlp']:
                total_angle_predictions = 0
                accurate_2deg_predictions = 0
                accurate_4deg_predictions = 0
                
                for angle_name in angle_names:
                    error_col = f'{model_name}_{angle_name}_error'
                    if error_col in angle_df.columns:
                        errors = angle_df[error_col].dropna()
                        if len(errors) > 0:
                            accurate_2deg_count = (errors <= 2.0).sum()
                            accurate_4deg_count = (errors <= 4.0).sum()
                            total_count = len(errors)
                            total_angle_predictions += total_count
                            accurate_2deg_predictions += accurate_2deg_count
                            accurate_4deg_predictions += accurate_4deg_count
                
                if total_angle_predictions > 0:
                    angle_accuracy_2deg = accurate_2deg_predictions / total_angle_predictions
                    angle_accuracy_4deg = accurate_4deg_predictions / total_angle_predictions
                    model_short = 'HRNetV2' if model_name == 'ensemble_hrnetv2' else 'MLP'
                    print(f"   📐 {model_short} Angle 2° Accuracy: {angle_accuracy_2deg:.1%}")
                    print(f"   📐 {model_short} Angle 4° Accuracy: {angle_accuracy_4deg:.1%}")
        except:
            pass
    
    if len(all_hrnet_preds) > 1 and args.evaluate_individual:
        individual_mres = [results[f'Model {i+1} MLP (Test)']['overall']['mre'] for i in range(len(all_hrnet_preds))]
        avg_individual_mre = np.mean(individual_mres)
        ensemble_improvement = (avg_individual_mre - ensemble_mre) / avg_individual_mre * 100
        print(f"📈 Improvement over average individual: {ensemble_improvement:+.1f}%")
    
    # Sella-specific summary
    if 'sella' in landmark_names:
        # Test set sella performance
        test_sella_results = []
        for name, metrics in results.items():
            if 'MLP' in name and 'sella' in metrics['per_landmark']:
                sella_metrics = metrics['per_landmark']['sella']
                mre_px = sella_metrics.get('mre', 0)
                mre_mm = sella_metrics.get('mre_mm', None)
                test_sella_results.append((name, mre_px, mre_mm))
        
        print(f"\n🎯 SELLA LANDMARK PERFORMANCE (TEST SET):")
        for model_name, sella_mre_px, sella_mre_mm in test_sella_results:
            if sella_mre_mm is not None:
                print(f"   {model_name}: {sella_mre_px:.3f} pixels / {sella_mre_mm:.3f} mm")
            else:
                print(f"   {model_name}: {sella_mre_px:.3f} pixels")
        
        # Validation set sella performance if available
        if validation_results:
            val_sella_results = [(name, metrics['per_landmark'].get('sella', {}).get('mre', 0)) 
                                for name, metrics in validation_results.items() if 'MLP' in name]
            
            if val_sella_results:
                print(f"\n🎯 SELLA LANDMARK PERFORMANCE (VALIDATION SET):")
                for model_name, sella_mre in val_sella_results:
                    print(f"   {model_name}: {sella_mre:.3f} pixels")

if __name__ == "__main__":
    main() 