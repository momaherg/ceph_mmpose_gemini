#!/usr/bin/env python3
"""
Diagnostic script to identify classification issues in HRNetV2 with native classification.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, os.getcwd())

# Import necessary modules
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.apis import init_model
import anb_classification_utils
from cephalometric_dataset_info import landmark_names_in_order, original_landmark_cols


def check_dataset_distribution(data_file):
    """Check the class distribution in the dataset."""
    print("\n" + "="*60)
    print("CHECKING DATASET CLASS DISTRIBUTION")
    print("="*60)
    
    # Load the dataset
    df = pd.read_json(data_file)
    print(f"Total samples: {len(df)}")
    
    # Check if 'class' column exists
    if 'class' in df.columns:
        print("\nClass distribution from 'class' column:")
        class_counts = df['class'].value_counts()
        print(class_counts)
        print(f"\nClass proportions:")
        print(class_counts / len(df))
    
    # Compute classification from ANB angles
    computed_classes = []
    valid_samples = 0
    
    for idx, row in df.iterrows():
        # Get landmarks for ANB calculation
        keypoints = []
        valid = True
        
        for i in range(0, len(original_landmark_cols), 2):
            x_col = original_landmark_cols[i]
            y_col = original_landmark_cols[i+1]
            
            if x_col in row and y_col in row and pd.notna(row[x_col]) and pd.notna(row[y_col]):
                keypoints.append([row[x_col], row[y_col]])
            else:
                keypoints.append([0, 0])
                # Check if this is a critical landmark for ANB (Nasion, A-point, B-point)
                if i//2 in [1, 2, 3]:  # Indices for nasion, A-point, B-point
                    valid = False
        
        if valid:
            keypoints = np.array(keypoints).reshape(1, 19, 2)
            anb_angle = anb_classification_utils.calculate_anb_angle(keypoints)
            class_label = anb_classification_utils.classify_from_anb_angle(anb_angle).item()
            computed_classes.append(class_label)
            valid_samples += 1
    
    print(f"\n\nComputed class distribution from ANB angles ({valid_samples} valid samples):")
    class_counter = Counter(computed_classes)
    for class_id in sorted(class_counter.keys()):
        count = class_counter[class_id]
        class_name = anb_classification_utils.get_class_name(class_id)
        print(f"Class {class_id} ({class_name}): {count} samples ({count/valid_samples*100:.1f}%)")
    
    return class_counter, valid_samples


def check_model_initialization(config_path, checkpoint_path=None):
    """Check model initialization and weights."""
    print("\n" + "="*60)
    print("CHECKING MODEL INITIALIZATION")
    print("="*60)
    
    # Initialize model
    model = init_model(config_path, checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check classification head weights
    if hasattr(model, 'head') and hasattr(model.head, 'classification_head'):
        print("\nClassification head architecture:")
        for name, module in model.head.classification_head.named_modules():
            if isinstance(module, torch.nn.Linear):
                print(f"{name}: Linear({module.in_features} -> {module.out_features})")
                
                # Check weight statistics
                weight_mean = module.weight.data.mean().item()
                weight_std = module.weight.data.std().item()
                bias_mean = module.bias.data.mean().item() if module.bias is not None else 0
                
                print(f"  Weight: mean={weight_mean:.4f}, std={weight_std:.4f}")
                print(f"  Bias: mean={bias_mean:.4f}")
                
                # Check if final layer bias is biased toward class 1
                if module.out_features == 3:  # Final classification layer
                    print(f"  Final layer biases: {module.bias.data.cpu().numpy()}")
                    max_bias_idx = torch.argmax(module.bias.data).item()
                    print(f"  Maximum bias at index: {max_bias_idx} (class {max_bias_idx})")
    
    return model


def test_classification_head(model, config_path):
    """Test the classification head with synthetic data."""
    print("\n" + "="*60)
    print("TESTING CLASSIFICATION HEAD")
    print("="*60)
    
    device = next(model.parameters()).device
    
    # Create synthetic features with different patterns
    batch_size = 10
    in_channels = 270  # HRNet concatenated features
    feat_h, feat_w = 56, 56  # Typical feature map size
    
    # Test 1: Random features
    random_feat = torch.randn(batch_size, in_channels, feat_h, feat_w).to(device)
    
    # Test 2: All zeros
    zero_feat = torch.zeros(batch_size, in_channels, feat_h, feat_w).to(device)
    
    # Test 3: All ones  
    ones_feat = torch.ones(batch_size, in_channels, feat_h, feat_w).to(device)
    
    test_features = {
        'random': random_feat,
        'zeros': zero_feat,
        'ones': ones_feat
    }
    
    model.eval()
    with torch.no_grad():
        for feat_name, feat in test_features.items():
            # Run through classification head
            if hasattr(model.head, 'classification_head'):
                logits = model.head.classification_head(feat)
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probs, dim=-1)
                
                print(f"\n{feat_name.upper()} features:")
                print(f"  Logits shape: {logits.shape}")
                print(f"  Mean logits per class: {logits.mean(dim=0).cpu().numpy()}")
                print(f"  Predictions: {predictions.cpu().numpy()}")
                print(f"  Prediction distribution: {Counter(predictions.cpu().numpy().tolist())}")
                print(f"  Average probabilities per class: {probs.mean(dim=0).cpu().numpy()}")


def analyze_training_gradients(config_path):
    """Analyze gradient flow during training."""
    print("\n" + "="*60)
    print("SUGGESTED FIXES")
    print("="*60)
    
    print("\n1. INCREASE CLASSIFICATION LOSS WEIGHT:")
    print("   Current: classification_loss_weight=0.5")
    print("   Suggested: classification_loss_weight=2.0 or higher")
    print("   The keypoint loss might be dominating training.")
    
    print("\n2. USE BALANCED CLASS WEIGHTS:")
    print("   Add class weights to handle imbalanced dataset:")
    print("   ```python")
    print("   # In model config")
    print("   classification_loss=dict(")
    print("       type='CrossEntropyLoss',")
    print("       weight=torch.tensor([1.5, 1.0, 1.5])  # Adjust based on class distribution")
    print("   )")
    print("   ```")
    
    print("\n3. SEPARATE LEARNING RATES:")
    print("   Use different learning rate for classification head:")
    print("   ```python")
    print("   # In optimizer config")
    print("   optim_wrapper = dict(")
    print("       paramwise_cfg=dict(")
    print("           custom_keys={")
    print("               'head.classification_head': dict(lr_mult=10.0),")
    print("           }")
    print("       )")
    print("   )")
    print("   ```")
    
    print("\n4. ADD GRADIENT MONITORING:")
    print("   Monitor classification loss separately during training")
    print("   to ensure it's decreasing.")
    
    print("\n5. FEATURE ADAPTATION LAYER:")
    print("   Add an adaptation layer before classification:")
    print("   ```python")
    print("   self.feature_adapter = nn.Sequential(")
    print("       nn.Conv2d(in_channels, 128, 1),")
    print("       nn.BatchNorm2d(128),")
    print("       nn.ReLU(inplace=True)")
    print("   )")
    print("   ```")


def main():
    """Main diagnostic function."""
    print("="*60)
    print("CLASSIFICATION ISSUE DIAGNOSTICS")
    print("="*60)
    
    # Configuration
    data_file = "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json"
    config_path = "Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py"
    
    # You can specify a checkpoint path if you want to test a trained model
    checkpoint_path = None  # or "work_dirs/path/to/checkpoint.pth"
    
    # Initialize MMPose
    init_default_scope('mmpose')
    
    # Import custom modules
    import custom_cephalometric_dataset
    import custom_transforms
    import cephalometric_dataset_info
    import hrnetv2_with_classification_simple
    import classification_evaluator
    
    # 1. Check dataset distribution
    class_counts, valid_samples = check_dataset_distribution(data_file)
    
    # Calculate class imbalance ratio
    if class_counts:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 2:
            print("⚠️  WARNING: Dataset is imbalanced! This can cause the model to predict majority class.")
    
    # 2. Check model initialization
    model = check_model_initialization(config_path, checkpoint_path)
    
    # 3. Test classification head
    test_classification_head(model, config_path)
    
    # 4. Provide suggested fixes
    analyze_training_gradients(config_path)
    
    # Calculate suggested class weights based on distribution
    if class_counts and len(class_counts) == 3:
        total_samples = sum(class_counts.values())
        class_weights = []
        for i in range(3):
            if i in class_counts and class_counts[i] > 0:
                # Inverse frequency weighting
                weight = total_samples / (3 * class_counts[i])
                class_weights.append(weight)
            else:
                class_weights.append(1.0)
        
        print(f"\n\nSUGGESTED CLASS WEIGHTS based on your dataset:")
        print(f"torch.tensor({[round(w, 2) for w in class_weights]})")


if __name__ == "__main__":
    main() 