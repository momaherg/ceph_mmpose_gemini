#!/usr/bin/env python3
"""
MLP-based Refinement Stage for Cephalometric Landmark Detection
This script implements a two-stage approach:
1. Generate training data by running HRNetV2 inference on all images
2. Train separate MLP models for x and y coordinates to refine predictions
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import warnings
import pandas as pd
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.apis import init_model, inference_topdown
import glob
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

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
    """
    MLP model for landmark coordinate refinement.
    Input: 19 predicted coordinates
    Hidden: 500 neurons
    Output: 19 refined coordinates
    """
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

class MLPDataset(data.Dataset):
    """Dataset for MLP training."""
    def __init__(self, predictions, ground_truth):
        self.predictions = torch.FloatTensor(predictions)
        self.ground_truth = torch.FloatTensor(ground_truth)
        
    def __len__(self):
        return len(self.predictions)
    
    def __getitem__(self, idx):
        return self.predictions[idx], self.ground_truth[idx]

def generate_mlp_training_data(args):
    """
    Stage 1: Generate training data by running HRNetV2 inference on all images.
    """
    print("="*80)
    print("STAGE 1: GENERATING MLP TRAINING DATA")
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
        return False
    
    # Load config
    try:
        cfg = Config.fromfile(args.hrnet_config)
        print(f"✓ Configuration loaded from {args.hrnet_config}")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return False
    
    # Find checkpoint if not provided
    if args.hrnet_checkpoint is None:
        work_dir = "work_dirs/hrnetv2_w18_cephalometric_384x384_adaptive_wing_loss_v4"
        checkpoint_pattern = os.path.join(work_dir, "best_NME_epoch_*.pth")
        checkpoints = glob.glob(checkpoint_pattern)
        
        if not checkpoints:
            checkpoint_pattern = os.path.join(work_dir, "epoch_*.pth")
            checkpoints = glob.glob(checkpoint_pattern)
        
        if not checkpoints:
            print(f"✗ No checkpoints found in {work_dir}")
            return False
        
        args.hrnet_checkpoint = max(checkpoints, key=os.path.getctime)
    
    print(f"✓ Using checkpoint: {args.hrnet_checkpoint}")
    
    # Load data
    try:
        main_df = pd.read_json(args.data_path)
        print(f"✓ Data loaded from {args.data_path}")
        print(f"  Dataset shape: {main_df.shape}")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return False
    
    # Split data following the same logic as train_improved_v4.py
    print(f"\n📊 Splitting data...")
    
    if args.test_split_file:
        print(f"Using external test set file: {args.test_split_file}")
        with open(args.test_split_file, 'r') as f:
            # Read IDs and convert to integer for matching
            test_patient_ids = {
                int(line.strip())
                for line in f if line.strip()
            }

        if 'patient_id' not in main_df.columns:
            print("ERROR: 'patient_id' column not found in the main DataFrame.")
            return False
        
        # Ensure the DataFrame's patient_id is also an integer
        main_df['patient_id'] = main_df['patient_id'].astype(int)

        test_df = main_df[main_df['patient_id'].isin(test_patient_ids)].reset_index(drop=True)
        remaining_df = main_df[~main_df['patient_id'].isin(test_patient_ids)]

        if len(remaining_df) >= 100:
            # We have enough data to sample 100 for validation
            val_df = remaining_df.sample(n=100, random_state=42)
            train_df = remaining_df.drop(val_df.index).reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
        else:
            # Not enough data for a 100-patient validation set
            print(f"WARNING: Only {len(remaining_df)} patients remaining after selecting the test set.")
            print("Splitting the remaining data into 50% validation and 50% training.")
            if len(remaining_df) > 1:
                val_df = remaining_df.sample(frac=0.5, random_state=42)
                train_df = remaining_df.drop(val_df.index).reset_index(drop=True)
                val_df = val_df.reset_index(drop=True)
            else:  # Only 0 or 1 patient left, not enough to split
                val_df = remaining_df.reset_index(drop=True)
                train_df = pd.DataFrame()  # Empty training set
    else:
        print("Splitting data using 'set' column from the JSON file.")
        train_df = main_df[main_df['set'] == 'train'].reset_index(drop=True)
        val_df = main_df[main_df['set'] == 'dev'].reset_index(drop=True)
        test_df = main_df[main_df['set'] == 'test'].reset_index(drop=True)

    print(f"✓ Data split completed:")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    if train_df.empty or val_df.empty:
        print("ERROR: Training or validation DataFrame is empty.")
        return False
    
    # Combine train and validation for MLP training (we'll split later in MLP training stage)
    mlp_source_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    print(f"✓ Using {len(mlp_source_df)} samples for MLP training (train + validation)")
    print(f"  Excluding {len(test_df)} test samples to prevent data leakage")
    
    # Get landmark information
    landmark_names = cephalometric_dataset_info.landmark_names_in_order
    landmark_cols = cephalometric_dataset_info.original_landmark_cols
    
    print(f"✓ Working with {len(landmark_names)} landmarks")
    
    # Initialize model
    try:
        model = init_model(args.hrnet_config, args.hrnet_checkpoint, 
                          device='cuda:0' if torch.cuda.is_available() else 'cpu')
        print("✓ HRNetV2 model initialized")
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        return False
    
    # Prepare data storage
    all_predictions_x = []
    all_predictions_y = []
    all_ground_truth_x = []
    all_ground_truth_y = []
    valid_samples = []
    original_sets = []  # Store original train/val labels
    
    print(f"\n🔄 Running inference on {len(mlp_source_df)} images (excluding test set)...")
    
    # Process each image
    for idx, row in mlp_source_df.iterrows():
        try:
            # Get image
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
            
            # Skip samples with invalid ground truth
            if not valid_gt:
                continue
                
            gt_keypoints = np.array(gt_keypoints)
            
            # Prepare bbox
            bbox = np.array([[0, 0, 224, 224]], dtype=np.float32)
            
            # Run inference
            results = inference_topdown(model, img_array, bboxes=bbox, bbox_format='xyxy')
            
            if results and len(results) > 0:
                pred_keypoints = results[0].pred_instances.keypoints[0]
                
                # Extract x and y coordinates
                pred_x = pred_keypoints[:, 0]
                pred_y = pred_keypoints[:, 1]
                gt_x = gt_keypoints[:, 0]
                gt_y = gt_keypoints[:, 1]
                
                # Store data
                all_predictions_x.append(pred_x)
                all_predictions_y.append(pred_y)
                all_ground_truth_x.append(gt_x)
                all_ground_truth_y.append(gt_y)
                valid_samples.append(idx)
                
                # Store original set information
                if args.test_split_file:
                    # Determine if this was originally train or val based on our split
                    original_idx = mlp_source_df.index[idx]
                    if original_idx in train_df.index:
                        original_sets.append('train')
                    else:
                        original_sets.append('val')
                else:
                    original_sets.append(row.get('set', 'unknown'))
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(mlp_source_df)} images")
                
        except Exception as e:
            print(f"  Warning: Failed to process image {idx}: {e}")
            continue
    
    print(f"✓ Successfully processed {len(valid_samples)} out of {len(mlp_source_df)} images")
    
    # Convert to numpy arrays
    all_predictions_x = np.array(all_predictions_x)
    all_predictions_y = np.array(all_predictions_y)
    all_ground_truth_x = np.array(all_ground_truth_x)
    all_ground_truth_y = np.array(all_ground_truth_y)
    
    # Create DataFrame for storage
    data_dict = {}
    
    # Add prediction coordinates
    for i in range(19):
        data_dict[f'pred_x_{i}'] = all_predictions_x[:, i]
        data_dict[f'pred_y_{i}'] = all_predictions_y[:, i]
        data_dict[f'gt_x_{i}'] = all_ground_truth_x[:, i]
        data_dict[f'gt_y_{i}'] = all_ground_truth_y[:, i]
    
    # Add metadata
    data_dict['sample_idx'] = valid_samples
    data_dict['original_set'] = original_sets
    
    mlp_df = pd.DataFrame(data_dict)
    
    # Save to CSV
    mlp_df.to_csv(args.mlp_data_path, index=False)
    print(f"✓ MLP training data saved to {args.mlp_data_path}")
    
    # Print statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total samples: {len(mlp_df)}")
    print(f"  Features per coordinate: 19 landmarks")
    print(f"  Input dimension: 19 (predicted coordinates)")
    print(f"  Output dimension: 19 (ground truth coordinates)")
    
    # Print distribution by original set
    set_counts = pd.Series(original_sets).value_counts()
    print(f"  Sample distribution:")
    for set_name, count in set_counts.items():
        print(f"    {set_name}: {count} samples")
    
    # Compute initial errors
    pred_coords = np.stack([all_predictions_x, all_predictions_y], axis=2)
    gt_coords = np.stack([all_ground_truth_x, all_ground_truth_y], axis=2)
    
    radial_errors = np.sqrt(np.sum((pred_coords - gt_coords)**2, axis=2))
    mean_errors = np.mean(radial_errors, axis=0)
    
    print(f"  Mean radial error per landmark (before refinement):")
    for i, (name, error) in enumerate(zip(landmark_names, mean_errors)):
        print(f"    {i:2d}. {name:<20}: {error:.3f} pixels")
    
    overall_mre = np.mean(radial_errors)
    print(f"  Overall MRE: {overall_mre:.3f} pixels")
    
    return True

def train_mlp_models(args):
    """
    Stage 2: Train separate MLP models for x and y coordinates.
    """
    print("\n" + "="*80)
    print("STAGE 2: TRAINING MLP REFINEMENT MODELS")
    print("="*80)
    
    # Load MLP training data
    try:
        mlp_df = pd.read_csv(args.mlp_data_path)
        print(f"✓ MLP training data loaded from {args.mlp_data_path}")
        print(f"  Dataset shape: {mlp_df.shape}")
    except Exception as e:
        print(f"✗ Failed to load MLP training data: {e}")
        return False
    
    # Prepare data
    pred_x_cols = [f'pred_x_{i}' for i in range(19)]
    pred_y_cols = [f'pred_y_{i}' for i in range(19)]
    gt_x_cols = [f'gt_x_{i}' for i in range(19)]
    gt_y_cols = [f'gt_y_{i}' for i in range(19)]
    
    # Extract features and targets
    X_x = mlp_df[pred_x_cols].values
    y_x = mlp_df[gt_x_cols].values
    X_y = mlp_df[pred_y_cols].values
    y_y = mlp_df[gt_y_cols].values
    
    print(f"✓ Data prepared:")
    print(f"  X coordinates - Input shape: {X_x.shape}, Target shape: {y_x.shape}")
    print(f"  Y coordinates - Input shape: {X_y.shape}, Target shape: {y_y.shape}")
    
    # Split data
    X_x_train, X_x_val, y_x_train, y_x_val = train_test_split(
        X_x, y_x, test_size=0.2, random_state=42)
    X_y_train, X_y_val, y_y_train, y_y_val = train_test_split(
        X_y, y_y, test_size=0.2, random_state=42)
    
    print(f"✓ Data split into train/validation:")
    print(f"  Training samples: {len(X_x_train)}")
    print(f"  Validation samples: {len(X_x_val)}")
    
    # Normalize data
    scaler_x_input = StandardScaler()
    scaler_x_target = StandardScaler()
    scaler_y_input = StandardScaler()
    scaler_y_target = StandardScaler()
    
    X_x_train_scaled = scaler_x_input.fit_transform(X_x_train)
    X_x_val_scaled = scaler_x_input.transform(X_x_val)
    y_x_train_scaled = scaler_x_target.fit_transform(y_x_train)
    y_x_val_scaled = scaler_x_target.transform(y_x_val)
    
    X_y_train_scaled = scaler_y_input.fit_transform(X_y_train)
    X_y_val_scaled = scaler_y_input.transform(X_y_val)
    y_y_train_scaled = scaler_y_target.fit_transform(y_y_train)
    y_y_val_scaled = scaler_y_target.transform(y_y_val)
    
    print("✓ Data normalized using StandardScaler")
    
    # Create datasets and dataloaders
    train_dataset_x = MLPDataset(X_x_train_scaled, y_x_train_scaled)
    val_dataset_x = MLPDataset(X_x_val_scaled, y_x_val_scaled)
    train_dataset_y = MLPDataset(X_y_train_scaled, y_y_train_scaled)
    val_dataset_y = MLPDataset(X_y_val_scaled, y_y_val_scaled)
    
    train_loader_x = data.DataLoader(train_dataset_x, batch_size=16, shuffle=True)
    val_loader_x = data.DataLoader(val_dataset_x, batch_size=16, shuffle=False)
    train_loader_y = data.DataLoader(train_dataset_y, batch_size=16, shuffle=True)
    val_loader_y = data.DataLoader(val_dataset_y, batch_size=16, shuffle=False)
    
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    # Training function
    def train_mlp(model, train_loader, val_loader, model_name, save_path):
        """Train a single MLP model."""
        print(f"\n🚀 Training {model_name} MLP...")
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(100):
            # Training
            model.train()
            epoch_train_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    epoch_val_loss += loss.item()
            
            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/100 - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        print(f"✓ {model_name} training completed. Best val loss: {best_val_loss:.6f}")
        return train_losses, val_losses
    
    # Initialize models
    model_x = MLPRefinementModel().to(device)
    model_y = MLPRefinementModel().to(device)
    
    print(f"✓ MLP models initialized:")
    print(f"  Architecture: 19 → 500 → 19")
    print(f"  Parameters per model: {sum(p.numel() for p in model_x.parameters()):,}")
    
    # Create output directory
    output_dir = "mlp_refinement_models"
    os.makedirs(output_dir, exist_ok=True)
    
    # Train X coordinate model
    model_x_path = os.path.join(output_dir, "mlp_x_model.pth")
    train_losses_x, val_losses_x = train_mlp(
        model_x, train_loader_x, val_loader_x, "X-coordinate", model_x_path)
    
    # Train Y coordinate model
    model_y_path = os.path.join(output_dir, "mlp_y_model.pth")
    train_losses_y, val_losses_y = train_mlp(
        model_y, train_loader_y, val_loader_y, "Y-coordinate", model_y_path)
    
    # Save scalers
    scaler_x_input_path = os.path.join(output_dir, "scaler_x_input.pkl")
    scaler_x_target_path = os.path.join(output_dir, "scaler_x_target.pkl")
    scaler_y_input_path = os.path.join(output_dir, "scaler_y_input.pkl")
    scaler_y_target_path = os.path.join(output_dir, "scaler_y_target.pkl")
    
    joblib.dump(scaler_x_input, scaler_x_input_path)
    joblib.dump(scaler_x_target, scaler_x_target_path)
    joblib.dump(scaler_y_input, scaler_y_input_path)
    joblib.dump(scaler_y_target, scaler_y_target_path)
    
    print(f"✓ Scalers saved to {output_dir}")
    
    # Plot training curves
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # X coordinate losses
    ax1.plot(train_losses_x, label='Training Loss', color='blue')
    ax1.plot(val_losses_x, label='Validation Loss', color='red')
    ax1.set_title('X-Coordinate MLP Training')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Y coordinate losses
    ax2.plot(train_losses_y, label='Training Loss', color='blue')
    ax2.plot(val_losses_y, label='Validation Loss', color='red')
    ax2.set_title('Y-Coordinate MLP Training')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Combined loss comparison
    ax3.plot(train_losses_x, label='X Train', color='blue', linestyle='-')
    ax3.plot(val_losses_x, label='X Val', color='blue', linestyle='--')
    ax3.plot(train_losses_y, label='Y Train', color='red', linestyle='-')
    ax3.plot(val_losses_y, label='Y Val', color='red', linestyle='--')
    ax3.set_title('Combined Training Comparison')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MSE Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Final loss values
    final_metrics = [
        ['X Train', train_losses_x[-1]],
        ['X Val', val_losses_x[-1]],
        ['Y Train', train_losses_y[-1]],
        ['Y Val', val_losses_y[-1]]
    ]
    
    metrics_df = pd.DataFrame(final_metrics, columns=['Model', 'Final Loss'])
    ax4.bar(metrics_df['Model'], metrics_df['Final Loss'], 
            color=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'])
    ax4.set_title('Final Training Metrics')
    ax4.set_ylabel('MSE Loss')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "mlp_training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training curves saved to {plot_path}")
    
    # Test refinement on validation set
    print(f"\n📊 Testing refinement on validation set...")
    
    # Load best models
    model_x.load_state_dict(torch.load(model_x_path))
    model_y.load_state_dict(torch.load(model_y_path))
    model_x.eval()
    model_y.eval()
    
    with torch.no_grad():
        # Get original predictions
        original_x = scaler_x_input.inverse_transform(X_x_val_scaled)
        original_y = scaler_y_input.inverse_transform(X_y_val_scaled)
        
        # Get refined predictions
        refined_x_scaled = model_x(torch.FloatTensor(X_x_val_scaled).to(device)).cpu().numpy()
        refined_y_scaled = model_y(torch.FloatTensor(X_y_val_scaled).to(device)).cpu().numpy()
        
        refined_x = scaler_x_target.inverse_transform(refined_x_scaled)
        refined_y = scaler_y_target.inverse_transform(refined_y_scaled)
        
        # Get ground truth
        gt_x = scaler_x_target.inverse_transform(y_x_val_scaled)
        gt_y = scaler_y_target.inverse_transform(y_y_val_scaled)
    
    # Compute errors
    original_coords = np.stack([original_x, original_y], axis=2)
    refined_coords = np.stack([refined_x, refined_y], axis=2)
    gt_coords = np.stack([gt_x, gt_y], axis=2)
    
    original_errors = np.sqrt(np.sum((original_coords - gt_coords)**2, axis=2))
    refined_errors = np.sqrt(np.sum((refined_coords - gt_coords)**2, axis=2))
    
    original_mre = np.mean(original_errors)
    refined_mre = np.mean(refined_errors)
    improvement = (original_mre - refined_mre) / original_mre * 100
    
    print(f"✓ Validation Results:")
    print(f"  Original MRE: {original_mre:.3f} pixels")
    print(f"  Refined MRE: {refined_mre:.3f} pixels")
    print(f"  Improvement: {improvement:.2f}%")
    
    # Save summary
    summary = {
        'stage': 'MLP Training Completed',
        'original_mre': original_mre,
        'refined_mre': refined_mre,
        'improvement_percent': improvement,
        'x_model_path': model_x_path,
        'y_model_path': model_y_path,
        'training_samples': len(X_x_train),
        'validation_samples': len(X_x_val)
    }
    
    summary_path = os.path.join(output_dir, "training_summary.json")
    pd.DataFrame([summary]).to_json(summary_path, indent=2)
    print(f"✓ Training summary saved to {summary_path}")
    
    return True

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='MLP-based Refinement Stage for Cephalometric Landmark Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate training data only:
  python train_mlp_refinement.py --generate-data
  
  # Train MLP models only (requires existing data):
  python train_mlp_refinement.py --train-mlp
  
  # Run both stages:
  python train_mlp_refinement.py --generate-data --train-mlp
        """)
    
    # --- Arguments for Data Generation ---
    gen_group = parser.add_argument_group('Data Generation')
    gen_group.add_argument('--generate-data', action='store_true', 
                          help='Run the data generation stage.')
    gen_group.add_argument('--hrnet-config', type=str, 
                          default='Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py', 
                          help='Path to the HRNetV2 config file.')
    gen_group.add_argument('--hrnet-checkpoint', type=str, default=None, 
                          help='Path to the HRNetV2 model checkpoint. If not given, finds the latest best model in work_dir.')
    gen_group.add_argument('--data-path', type=str, 
                          default="/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json", 
                          help='Path to the JSON file with image data and ground truth.')
    gen_group.add_argument('--test-split-file', type=str, default=None, 
                          help='Path to the external test set file.')
    
    # --- Arguments for MLP Training ---
    train_group = parser.add_argument_group('MLP Training')
    train_group.add_argument('--train-mlp', action='store_true', 
                           help='Run the MLP training stage.')
    
    # --- Shared Arguments ---
    parser.add_argument('--mlp-data-path', type=str, default='mlp_training_data.csv', 
                       help='Path to save/load the intermediate MLP training data CSV.')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.generate_data and not args.train_mlp:
        print("ERROR: Must specify at least one of --generate-data or --train-mlp")
        parser.print_help()
        return
    
    print("="*80)
    print("MLP-BASED REFINEMENT FOR CEPHALOMETRIC LANDMARK DETECTION")
    print("="*80)
    print("🎯 Goal: Train MLPs to refine HRNetV2 predictions")
    print("📊 Architecture: 19 → 500 → 19 (separate for X and Y)")
    print("⚙️  Training: 100 epochs, batch=16, lr=1e-5, Adam optimizer")
    print("="*80)
    
    success = True
    
    # Stage 1: Generate training data
    if args.generate_data:
        success = generate_mlp_training_data(args)
        if not success:
            print("💥 Data generation failed!")
            return
    
    # Stage 2: Train MLP models
    if args.train_mlp:
        if not os.path.exists(args.mlp_data_path):
            print(f"ERROR: MLP training data not found at {args.mlp_data_path}")
            print("Please run with --generate-data first")
            return
        
        success = train_mlp_models(args)
        if not success:
            print("💥 MLP training failed!")
            return
    
    if success:
        print("\n🎉 MLP refinement training completed successfully!")
        print("\n📋 Next steps:")
        print("1. 🧪 Test the refinement models on test set")
        print("2. 📊 Compare refined vs original predictions")
        print("3. 🎨 Visualize improvement on challenging landmarks")
        print("4. 🚀 Integrate refinement into inference pipeline")

if __name__ == "__main__":
    main() 