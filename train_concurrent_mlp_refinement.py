#!/usr/bin/env python3
"""
Concurrent MLP Refinement Training for Cephalometric Landmark Detection
This script implements a dynamic training strategy where:
1. HRNetV2 trains for one epoch
2. Current HRNetV2 generates predictions on training data
3. MLP trains for 100 epochs using those predictions
4. Cycle repeats for the entire HRNetV2 training duration

This allows the MLP to dynamically adapt to the evolving HRNetV2 model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import warnings
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.apis import init_model, inference_topdown
from mmengine.runner import Runner
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import StandardScaler
import joblib
import json
from collections import defaultdict
import copy

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

class ConcurrentMLPTrainer:
    """Handles the concurrent MLP training logic."""
    
    def __init__(self, device='cuda:0', mlp_epochs=100, lr=0.00001, weight_decay=0.0001):
        self.device = device
        self.mlp_epochs = mlp_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Initialize MLP models (only once!)
        self.model_x = MLPRefinementModel().to(device)
        self.model_y = MLPRefinementModel().to(device)
        
        # Initialize optimizers
        self.optimizer_x = optim.Adam(self.model_x.parameters(), lr=lr, weight_decay=weight_decay)
        self.optimizer_y = optim.Adam(self.model_y.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Initialize scalers (persistent across HRNetV2 epochs)
        self.scaler_x_input = StandardScaler()
        self.scaler_x_target = StandardScaler()
        self.scaler_y_input = StandardScaler()
        self.scaler_y_target = StandardScaler()
        self.scalers_fitted = False
        
        # Training history
        self.history = {
            'hrnet_epochs': [],
            'mlp_x_losses': [],
            'mlp_y_losses': [],
            'refinement_improvements': []
        }
        
        print(f"✓ Concurrent MLP Trainer initialized")
        print(f"  Device: {device}")
        print(f"  MLP epochs per HRNetV2 epoch: {mlp_epochs}")
        print(f"  MLP architecture: 19 → 500 → 19")
        print(f"  Parameters per MLP: {sum(p.numel() for p in self.model_x.parameters()):,}")
    
    def generate_predictions(self, hrnet_model, train_dataloader, landmark_cols):
        """Generate predictions using current HRNetV2 model."""
        hrnet_model.eval()  # Set to eval mode for inference
        
        all_predictions_x = []
        all_predictions_y = []
        all_ground_truth_x = []
        all_ground_truth_y = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(train_dataloader):
                try:
                    # Extract data samples from MMPose batch structure
                    data_samples = batch_data.get('data_samples', [])
                    inputs = batch_data.get('inputs', None)
                    
                    if inputs is None or len(data_samples) == 0:
                        continue
                    
                    # Process each sample in the batch
                    for sample_idx, data_sample in enumerate(data_samples):
                        try:
                            # Get ground truth keypoints
                            if hasattr(data_sample, 'gt_instances') and hasattr(data_sample.gt_instances, 'keypoints'):
                                gt_keypoints = data_sample.gt_instances.keypoints.numpy()
                                if gt_keypoints.shape[0] == 0:  # No keypoints
                                    continue
                                
                                # Get the image tensor
                                if sample_idx < len(inputs):
                                    img_tensor = inputs[sample_idx]
                                else:
                                    continue
                                
                                # Convert tensor to numpy for inference
                                if isinstance(img_tensor, torch.Tensor):
                                    # Handle different tensor formats
                                    if img_tensor.dim() == 3:  # C, H, W
                                        img_array = img_tensor.permute(1, 2, 0).cpu().numpy()
                                    else:  # Unexpected format
                                        continue
                                    
                                    # Denormalize if needed (assuming ImageNet normalization)
                                    if img_array.min() < 0:  # Likely normalized
                                        # Reverse ImageNet normalization
                                        mean = np.array([0.485, 0.456, 0.406])
                                        std = np.array([0.229, 0.224, 0.225])
                                        img_array = img_array * std + mean
                                    
                                    # Convert to uint8
                                    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
                                else:
                                    continue
                                
                                # Get image dimensions
                                h, w = img_array.shape[:2]
                                bbox = np.array([[0, 0, w, h]], dtype=np.float32)
                                
                                # Run inference using the HRNetV2 model
                                results = inference_topdown(hrnet_model, img_array, bboxes=bbox, bbox_format='xyxy')
                                
                                if results and len(results) > 0:
                                    pred_keypoints = results[0].pred_instances.keypoints[0]
                                    
                                    # Extract coordinates
                                    pred_x = pred_keypoints[:, 0]
                                    pred_y = pred_keypoints[:, 1]
                                    
                                    # Handle ground truth format (assuming first person, all keypoints)
                                    if gt_keypoints.ndim == 3:  # [num_persons, num_keypoints, 2/3]
                                        gt_x = gt_keypoints[0, :, 0]
                                        gt_y = gt_keypoints[0, :, 1]
                                    elif gt_keypoints.ndim == 2:  # [num_keypoints, 2/3]
                                        gt_x = gt_keypoints[:, 0]
                                        gt_y = gt_keypoints[:, 1]
                                    else:
                                        continue
                                    
                                    # Ensure we have 19 landmarks
                                    if len(pred_x) == 19 and len(gt_x) == 19:
                                        all_predictions_x.append(pred_x)
                                        all_predictions_y.append(pred_y)
                                        all_ground_truth_x.append(gt_x)
                                        all_ground_truth_y.append(gt_y)
                                
                        except Exception as e:
                            print(f"      Warning: Failed to process sample {sample_idx}: {e}")
                            continue
                
                except Exception as e:
                    print(f"    Warning: Failed to process batch {batch_idx}: {e}")
                    continue
                
                # Limit to reasonable number of samples per epoch
                if len(all_predictions_x) >= 500:
                    break
        
        print(f"    📊 Generated {len(all_predictions_x)} prediction samples for MLP training")
        
        if len(all_predictions_x) == 0:
            return None, None, None, None
        
        return (np.array(all_predictions_x), np.array(all_predictions_y),
                np.array(all_ground_truth_x), np.array(all_ground_truth_y))
    
    def prepare_mlp_data(self, pred_x, pred_y, gt_x, gt_y):
        """Prepare and normalize data for MLP training."""
        # Fit scalers on first epoch, then reuse
        if not self.scalers_fitted:
            self.scaler_x_input.fit(pred_x)
            self.scaler_x_target.fit(gt_x)
            self.scaler_y_input.fit(pred_y)
            self.scaler_y_target.fit(gt_y)
            self.scalers_fitted = True
            print(f"  ✓ MLP scalers fitted on first epoch data")
        
        # Transform data
        pred_x_scaled = self.scaler_x_input.transform(pred_x)
        pred_y_scaled = self.scaler_y_input.transform(pred_y)
        gt_x_scaled = self.scaler_x_target.transform(gt_x)
        gt_y_scaled = self.scaler_y_target.transform(gt_y)
        
        return pred_x_scaled, pred_y_scaled, gt_x_scaled, gt_y_scaled
    
    def train_mlp_cycle(self, pred_x_scaled, pred_y_scaled, gt_x_scaled, gt_y_scaled, hrnet_epoch):
        """Train MLP models for specified number of epochs."""
        print(f"    🤖 Training MLP for {self.mlp_epochs} epochs using HRNetV2 epoch {hrnet_epoch} predictions")
        
        # Create datasets
        dataset_x = MLPDataset(pred_x_scaled, gt_x_scaled)
        dataset_y = MLPDataset(pred_y_scaled, gt_y_scaled)
        
        # Create dataloaders
        dataloader_x = data.DataLoader(dataset_x, batch_size=16, shuffle=True)
        dataloader_y = data.DataLoader(dataset_y, batch_size=16, shuffle=True)
        
        # Train X coordinate model
        self.model_x.train()
        x_losses = []
        for epoch in range(self.mlp_epochs):
            epoch_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(dataloader_x):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer_x.zero_grad()
                outputs = self.model_x(inputs)
                loss = nn.MSELoss()(outputs, targets)
                loss.backward()
                self.optimizer_x.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader_x)
            x_losses.append(avg_loss)
        
        # Train Y coordinate model
        self.model_y.train()
        y_losses = []
        for epoch in range(self.mlp_epochs):
            epoch_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(dataloader_y):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer_y.zero_grad()
                outputs = self.model_y(inputs)
                loss = nn.MSELoss()(outputs, targets)
                loss.backward()
                self.optimizer_y.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader_y)
            y_losses.append(avg_loss)
        
        final_x_loss = x_losses[-1]
        final_y_loss = y_losses[-1]
        
        print(f"    ✓ MLP training completed - X loss: {final_x_loss:.6f}, Y loss: {final_y_loss:.6f}")
        
        # Store history
        self.history['hrnet_epochs'].append(hrnet_epoch)
        self.history['mlp_x_losses'].append(final_x_loss)
        self.history['mlp_y_losses'].append(final_y_loss)
        
        return final_x_loss, final_y_loss
    
    def evaluate_refinement(self, pred_x_scaled, pred_y_scaled, gt_x_scaled, gt_y_scaled):
        """Evaluate refinement improvement."""
        self.model_x.eval()
        self.model_y.eval()
        
        with torch.no_grad():
            # Get refined predictions
            pred_x_tensor = torch.FloatTensor(pred_x_scaled).to(self.device)
            pred_y_tensor = torch.FloatTensor(pred_y_scaled).to(self.device)
            
            refined_x_scaled = self.model_x(pred_x_tensor).cpu().numpy()
            refined_y_scaled = self.model_y(pred_y_tensor).cpu().numpy()
            
            # Inverse transform to original scale
            original_pred_x = self.scaler_x_input.inverse_transform(pred_x_scaled)
            original_pred_y = self.scaler_y_input.inverse_transform(pred_y_scaled)
            refined_x = self.scaler_x_target.inverse_transform(refined_x_scaled)
            refined_y = self.scaler_y_target.inverse_transform(refined_y_scaled)
            gt_x = self.scaler_x_target.inverse_transform(gt_x_scaled)
            gt_y = self.scaler_y_target.inverse_transform(gt_y_scaled)
        
        # Compute errors
        original_coords = np.stack([original_pred_x, original_pred_y], axis=2)
        refined_coords = np.stack([refined_x, refined_y], axis=2)
        gt_coords = np.stack([gt_x, gt_y], axis=2)
        
        original_errors = np.sqrt(np.sum((original_coords - gt_coords)**2, axis=2))
        refined_errors = np.sqrt(np.sum((refined_coords - gt_coords)**2, axis=2))
        
        original_mre = np.mean(original_errors)
        refined_mre = np.mean(refined_errors)
        improvement = (original_mre - refined_mre) / original_mre * 100
        
        self.history['refinement_improvements'].append(improvement)
        
        return original_mre, refined_mre, improvement
    
    def save_models(self, work_dir, hrnet_epoch):
        """Save current MLP models and scalers."""
        mlp_dir = os.path.join(work_dir, "concurrent_mlp_models")
        os.makedirs(mlp_dir, exist_ok=True)
        
        # Save models
        torch.save(self.model_x.state_dict(), os.path.join(mlp_dir, f"mlp_x_epoch_{hrnet_epoch}.pth"))
        torch.save(self.model_y.state_dict(), os.path.join(mlp_dir, f"mlp_y_epoch_{hrnet_epoch}.pth"))
        
        # Save scalers
        joblib.dump(self.scaler_x_input, os.path.join(mlp_dir, "scaler_x_input.pkl"))
        joblib.dump(self.scaler_x_target, os.path.join(mlp_dir, "scaler_x_target.pkl"))
        joblib.dump(self.scaler_y_input, os.path.join(mlp_dir, "scaler_y_input.pkl"))
        joblib.dump(self.scaler_y_target, os.path.join(mlp_dir, "scaler_y_target.pkl"))
        
        # Save training history
        with open(os.path.join(mlp_dir, "training_history.json"), 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_training_progress(self, work_dir):
        """Plot concurrent training progress."""
        mlp_dir = os.path.join(work_dir, "concurrent_mlp_models")
        
        if len(self.history['hrnet_epochs']) < 2:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = self.history['hrnet_epochs']
        
        # MLP loss evolution
        ax1.plot(epochs, self.history['mlp_x_losses'], 'b-', label='X Coordinate Loss')
        ax1.plot(epochs, self.history['mlp_y_losses'], 'r-', label='Y Coordinate Loss')
        ax1.set_xlabel('HRNetV2 Epoch')
        ax1.set_ylabel('MLP Loss')
        ax1.set_title('MLP Loss Evolution During Concurrent Training')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Refinement improvement
        ax2.plot(epochs, self.history['refinement_improvements'], 'g-', marker='o')
        ax2.set_xlabel('HRNetV2 Epoch')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('Refinement Improvement Over HRNetV2')
        ax2.grid(True, alpha=0.3)
        
        # Combined loss trend
        combined_loss = [(x + y) / 2 for x, y in zip(self.history['mlp_x_losses'], self.history['mlp_y_losses'])]
        ax3.plot(epochs, combined_loss, 'm-', linewidth=2)
        ax3.set_xlabel('HRNetV2 Epoch')
        ax3.set_ylabel('Average MLP Loss')
        ax3.set_title('Overall MLP Learning Trend')
        ax3.grid(True, alpha=0.3)
        
        # Improvement distribution
        improvements = self.history['refinement_improvements']
        ax4.hist(improvements, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(np.mean(improvements), color='red', linestyle='--', label=f'Mean: {np.mean(improvements):.2f}%')
        ax4.set_xlabel('Improvement (%)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Refinement Improvements')
        ax4.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(mlp_dir, "concurrent_training_progress.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    📊 Training progress plot saved to {plot_path}")

class CustomRunner(Runner):
    """Custom Runner that integrates MLP training after each epoch."""
    
    def __init__(self, *args, **kwargs):
        # Extract MLP trainer from kwargs before calling super()
        self.mlp_trainer = kwargs.pop('mlp_trainer', None)
        self.landmark_cols = kwargs.pop('landmark_cols', None)
        
        super().__init__(*args, **kwargs)
        
        if self.mlp_trainer is None:
            raise ValueError("MLP trainer must be provided")
    
    @classmethod
    def from_cfg(cls, cfg, **kwargs):
        """Override from_cfg to handle custom arguments."""
        # Extract custom arguments
        mlp_trainer = kwargs.pop('mlp_trainer', None)
        landmark_cols = kwargs.pop('landmark_cols', None)
        
        # Create runner normally
        runner = super().from_cfg(cfg, **kwargs)
        
        # Set custom attributes
        runner.mlp_trainer = mlp_trainer
        runner.landmark_cols = landmark_cols
        
        if runner.mlp_trainer is None:
            raise ValueError("MLP trainer must be provided")
        
        return runner
    
    def train_epoch(self):
        """Override train_epoch to include MLP training."""
        # Standard HRNetV2 training for one epoch
        super().train_epoch()
        
        # Get current epoch
        current_epoch = self._epoch
        
        print(f"\n  🔄 Starting concurrent MLP training after HRNetV2 epoch {current_epoch}")
        
        # Generate predictions using current HRNetV2 model
        pred_x, pred_y, gt_x, gt_y = self.mlp_trainer.generate_predictions(
            self.model, self.train_dataloader, self.landmark_cols)
        
        if pred_x is not None:
            # Prepare MLP data
            pred_x_scaled, pred_y_scaled, gt_x_scaled, gt_y_scaled = self.mlp_trainer.prepare_mlp_data(
                pred_x, pred_y, gt_x, gt_y)
            
            # Train MLP
            mlp_x_loss, mlp_y_loss = self.mlp_trainer.train_mlp_cycle(
                pred_x_scaled, pred_y_scaled, gt_x_scaled, gt_y_scaled, current_epoch)
            
            # Evaluate refinement
            original_mre, refined_mre, improvement = self.mlp_trainer.evaluate_refinement(
                pred_x_scaled, pred_y_scaled, gt_x_scaled, gt_y_scaled)
            
            print(f"  📈 Refinement: {original_mre:.3f} → {refined_mre:.3f} pixels ({improvement:+.2f}%)")
            
            # Save models periodically
            if current_epoch % 10 == 0:
                self.mlp_trainer.save_models(self.work_dir, current_epoch)
                self.mlp_trainer.plot_training_progress(self.work_dir)
        else:
            print(f"  ⚠️  No valid predictions generated for MLP training at epoch {current_epoch}")

def main():
    """Main concurrent training function."""
    
    parser = argparse.ArgumentParser(
        description='Concurrent MLP Refinement Training for Cephalometric Landmark Detection')
    parser.add_argument(
        '--test_split_file',
        type=str,
        default=None,
        help='Path to a text file containing patient IDs for the test set, one ID per line.'
    )
    parser.add_argument(
        '--mlp_epochs',
        type=int,
        default=100,
        help='Number of MLP training epochs per HRNetV2 epoch (default: 100)'
    )
    args = parser.parse_args()
    
    print("="*80)
    print("CONCURRENT MLP REFINEMENT TRAINING")
    print("="*80)
    print("🔄 Dynamic training strategy:")
    print("   1. HRNetV2 trains for 1 epoch")
    print("   2. Current HRNetV2 generates predictions")
    print(f"   3. MLP trains for {args.mlp_epochs} epochs")
    print("   4. Cycle repeats throughout training")
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
    work_dir = "work_dirs/hrnetv2_w18_cephalometric_concurrent_mlp_refinement"
    
    print(f"Config: {config_path}")
    print(f"Work Dir: {work_dir}")
    
    # Load config
    try:
        cfg = Config.fromfile(config_path)
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return
    
    # Set work directory
    cfg.work_dir = os.path.abspath(work_dir)
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # Load and prepare data (same logic as train_improved_v4.py)
    data_file_path = "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json"
    print(f"Loading main data file from: {data_file_path}")
    
    try:
        main_df = pd.read_json(data_file_path)
        print(f"Main DataFrame loaded. Shape: {main_df.shape}")

        if args.test_split_file:
            print(f"Splitting data using external test set file: {args.test_split_file}")
            with open(args.test_split_file, 'r') as f:
                test_patient_ids = {int(line.strip()) for line in f if line.strip()}

            if 'patient_id' not in main_df.columns:
                print("ERROR: 'patient_id' column not found in the main DataFrame.")
                return
            
            main_df['patient_id'] = main_df['patient_id'].astype(int)
            test_df = main_df[main_df['patient_id'].isin(test_patient_ids)].reset_index(drop=True)
            remaining_df = main_df[~main_df['patient_id'].isin(test_patient_ids)]

            if len(remaining_df) >= 100:
                val_df = remaining_df.sample(n=100, random_state=42)
                train_df = remaining_df.drop(val_df.index).reset_index(drop=True)
                val_df = val_df.reset_index(drop=True)
            else:
                print(f"WARNING: Only {len(remaining_df)} patients remaining after selecting the test set.")
                if len(remaining_df) > 1:
                    val_df = remaining_df.sample(frac=0.5, random_state=42)
                    train_df = remaining_df.drop(val_df.index).reset_index(drop=True)
                    val_df = val_df.reset_index(drop=True)
                else:
                    val_df = remaining_df.reset_index(drop=True)
                    train_df = pd.DataFrame()
        else:
            print("Splitting data using 'set' column from the JSON file.")
            train_df = main_df[main_df['set'] == 'train'].reset_index(drop=True)
            val_df = main_df[main_df['set'] == 'dev'].reset_index(drop=True)
            test_df = main_df[main_df['set'] == 'test'].reset_index(drop=True)

        print(f"Train DataFrame shape: {train_df.shape}")
        print(f"Validation DataFrame shape: {val_df.shape}")
        print(f"Test DataFrame shape: {test_df.shape}")

        if train_df.empty or val_df.empty:
            print("ERROR: Training or validation DataFrame is empty.")
            return

        # Data processing and balancing (same as train_improved_v4.py)
        def _angle(p1, p2, p3):
            v1 = np.array(p1) - np.array(p2)
            v2 = np.array(p3) - np.array(p2)
            norm_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm_prod == 0:
                return 0.0
            cos_val = np.clip(np.dot(v1, v2) / norm_prod, -1.0, 1.0)
            return np.degrees(np.arccos(cos_val))

        def _compute_class(row):
            if 'class' in row and pd.notna(row['class']):
                val = str(row['class']).strip().upper()
                if val in ['1', 'I']:
                    return 1
                if val in ['2', 'II']:
                    return 2
                if val in ['3', 'III']:
                    return 3

            s = (row['sella_x'], row['sella_y'])
            n = (row['nasion_x'], row['nasion_y'])
            a_pt = (row['A point_x'], row['A point_y'])
            b_pt = (row['B point_x'], row['B point_y'])

            sna = _angle(s, n, a_pt)
            snb = _angle(s, n, b_pt)
            anb = sna - snb

            if anb > 4:
                return 2
            elif anb < 2:
                return 3
            else:
                return 1

        if 'class' not in train_df.columns:
            train_df['class'] = np.nan
        train_df['class'] = train_df.apply(_compute_class, axis=1)

        # Balance training set
        print("Balancing training set class distribution:")
        print(train_df['class'].value_counts())
        class_counts = train_df['class'].value_counts()
        max_count = class_counts.max()

        balanced_parts = []
        for cls, subset in train_df.groupby('class'):
            balanced_subset = subset.sample(max_count, replace=True, random_state=42)
            balanced_parts.append(balanced_subset)

        train_df = pd.concat(balanced_parts).reset_index(drop=True)
        print("Balanced training set class distribution:")
        print(train_df['class'].value_counts())

        # Save DataFrames to temporary JSON files
        temp_train_ann_file = os.path.join(cfg.work_dir, 'temp_train_ann.json')
        temp_val_ann_file = os.path.join(cfg.work_dir, 'temp_val_ann.json')
        temp_test_ann_file = os.path.join(cfg.work_dir, 'temp_test_ann.json')

        train_df.to_json(temp_train_ann_file, orient='records', indent=2)
        val_df.to_json(temp_val_ann_file, orient='records', indent=2)
        if not test_df.empty:
            test_df.to_json(temp_test_ann_file, orient='records', indent=2)

        # Update config
        cfg.train_dataloader.dataset.ann_file = temp_train_ann_file
        cfg.train_dataloader.dataset.data_df = None
        cfg.train_dataloader.dataset.data_root = ''

        cfg.val_dataloader.dataset.ann_file = temp_val_ann_file
        cfg.val_dataloader.dataset.data_df = None
        cfg.val_dataloader.dataset.data_root = ''

        if not test_df.empty:
            cfg.test_dataloader.dataset.ann_file = temp_test_ann_file
        else:
            cfg.test_dataloader.dataset.ann_file = temp_val_ann_file
        cfg.test_dataloader.dataset.data_df = None
        cfg.test_dataloader.dataset.data_root = ''
        
        print("✓ Configuration updated with data files.")

    except Exception as e:
        print(f"ERROR: Failed to load or process data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize concurrent MLP trainer
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    mlp_trainer = ConcurrentMLPTrainer(
        device=device,
        mlp_epochs=args.mlp_epochs,
        lr=0.00001,
        weight_decay=0.0001
    )
    
    # Get landmark information
    landmark_cols = cephalometric_dataset_info.original_landmark_cols
    
    print("\n" + "="*70)
    print("🚀 STARTING CONCURRENT TRAINING")
    print("="*70)
    print(f"🎯 HRNetV2 will train normally")
    print(f"🤖 After each epoch, MLP trains for {args.mlp_epochs} epochs")
    print(f"🔄 This creates a dynamic adaptation process")
    print(f"📈 MLP learns to refine the evolving HRNetV2 predictions")
    
    try:
        # Build custom runner with MLP integration
        runner = CustomRunner.from_cfg(
            cfg, 
            mlp_trainer=mlp_trainer, 
            landmark_cols=landmark_cols
        )
        
        print("\n⏱️  Starting concurrent training...")
        runner.train()
        
        print("\n🎉 Concurrent training completed successfully!")
        
        # Final model saves
        mlp_trainer.save_models(cfg.work_dir, cfg.train_cfg.max_epochs)
        mlp_trainer.plot_training_progress(cfg.work_dir)
        
    except Exception as e:
        print(f"\n💥 Concurrent training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*70)
    print("🏆 CONCURRENT TRAINING COMPLETED")
    print("="*70)
    print("📊 Results:")
    if len(mlp_trainer.history['refinement_improvements']) > 0:
        improvements = mlp_trainer.history['refinement_improvements']
        print(f"  Average refinement improvement: {np.mean(improvements):.2f}%")
        print(f"  Best refinement improvement: {np.max(improvements):.2f}%")
        print(f"  Final refinement improvement: {improvements[-1]:.2f}%")
    
    print(f"\n📁 Models saved in: {cfg.work_dir}/concurrent_mlp_models/")
    print("🎯 Next steps:")
    print("1. 🧪 Evaluate the final concurrent model on test set")
    print("2. 📊 Compare against standalone HRNetV2 and static MLP")
    print("3. 🎨 Analyze the dynamic adaptation behavior")

if __name__ == "__main__":
    main() 