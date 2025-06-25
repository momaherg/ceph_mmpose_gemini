#!/usr/bin/env python3
"""
MLP Refinement Stage for HRNetV2 Cephalometric Landmark Detection
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import argparse
import glob
from tqdm import tqdm

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.dataset import Compose

# Import custom modules from the workspace
try:
    import custom_cephalometric_dataset
    import custom_transforms
    import cephalometric_dataset_info
    from mmpose.apis import init_model, inference_topdown
    print("✓ Custom modules and MMPose APIs imported successfully")
except ImportError as e:
    print(f"✗ Failed to import custom modules or MMPose: {e}")
    exit()

def generate_mlp_training_data(args):
    """
    Runs inference with a trained HRNetV2 model to generate a dataset
    of (predicted_landmarks, ground_truth_landmarks) pairs.
    """
    print("="*80)
    print("🚀 Stage 1: Generating MLP Training Data")
    print("="*80)

    # Load MMPose config and checkpoint
    print(f"Loading config: {args.hrnet_config}")
    cfg = Config.fromfile(args.hrnet_config)

    if not args.hrnet_checkpoint:
        # If checkpoint is not provided, find the latest 'best' checkpoint
        checkpoint_pattern = os.path.join(cfg.work_dir, "best_NME_epoch_*.pth")
        checkpoint_files = sorted(glob.glob(checkpoint_pattern), key=os.path.getmtime, reverse=True)
        if not checkpoint_files:
            print(f"✗ ERROR: No 'best' checkpoint found in {cfg.work_dir}. Please specify with --hrnet_checkpoint.")
            return
        args.hrnet_checkpoint = checkpoint_files[0]
        print(f"✓ Found latest best checkpoint: {args.hrnet_checkpoint}")

    # Initialize the HRNet model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = init_model(cfg, args.hrnet_checkpoint, device=device)
    
    # Prepare dataset
    print(f"Loading main data file from: {args.data_path}")
    full_df = pd.read_json(args.data_path)
    
    # We use the validation pipeline for inference to avoid augmentations
    # but ensure the input is processed correctly for the model.
    pipeline = Compose(cfg.val_pipeline)
    
    all_preds = []
    all_gts = []
    
    print("🏃‍♂️ Running inference on all images...")
    for i, row in tqdm(full_df.iterrows(), total=len(full_df)):
        # Construct the data sample for the inference pipeline
        data_info = {
            'img_array': row['img_array'],
            'bbox': np.array([[0, 0, row['img_cols'], row['img_rows']]], dtype=np.float32),
            'keypoints': np.array([row['landmarks']], dtype=np.float32)
        }
        
        # Apply the pipeline transformations
        processed_data = pipeline(data_info)
        
        # Run inference
        results = inference_topdown(model, processed_data['img_array'])
        
        pred_keypoints = results[0].pred_instances.keypoints[0] # Shape: (19, 2)
        gt_keypoints = data_info['keypoints'][0] # Shape: (19, 2) (use original, non-transformed GT)

        all_preds.append(pred_keypoints)
        all_gts.append(gt_keypoints)

    # Flatten the data for the MLP
    preds_flat = np.array(all_preds).reshape(len(all_preds), -1)
    gts_flat = np.array(all_gts).reshape(len(all_gts), -1)

    # Create a DataFrame
    columns = []
    for i in range(19):
        columns.append(f'pred_x_{i}')
        columns.append(f'pred_y_{i}')
    for i in range(19):
        columns.append(f'gt_x_{i}')
        columns.append(f'gt_y_{i}')

    data_for_df = np.hstack((preds_flat, gts_flat))
    df = pd.DataFrame(data_for_df, columns=columns)
    
    # Save to CSV
    df.to_csv(args.mlp_data_path, index=False)
    print(f"\n✅ Successfully generated and saved MLP training data to: {args.mlp_data_path}")


class LandmarkDataset(Dataset):
    """PyTorch Dataset for MLP landmark refinement."""
    def __init__(self, csv_file, coord_type='x'):
        self.data = pd.read_csv(csv_file)
        self.coord_type = coord_type
        
        pred_cols = [f'pred_{coord_type}_{i}' for i in range(19)]
        gt_cols = [f'gt_{coord_type}_{i}' for i in range(19)]
        
        self.preds = torch.tensor(self.data[pred_cols].values, dtype=torch.float32)
        self.gts = torch.tensor(self.data[gt_cols].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.preds[idx], self.gts[idx]


class MLP(nn.Module):
    """A simple MLP with one hidden layer."""
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(19, 500),
            nn.ReLU(),
            nn.Linear(500, 19)
        )

    def forward(self, x):
        return self.model(x)

def train_mlp_model(args, coord_type):
    """Trains a single MLP model for either 'x' or 'y' coordinates."""
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"\n--- Training MLP for {coord_type.upper()}-coordinates on {device} ---")

    # Data
    full_dataset = LandmarkDataset(args.mlp_data_path, coord_type=coord_type)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Model, Loss, Optimizer
    model = MLP().to(device)
    criterion = nn.MSELoss() # L2 Loss
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)

    best_val_loss = float('inf')

    for epoch in range(100):
        model.train()
        train_loss = 0.0
        for preds, gts in train_loader:
            preds, gts = preds.to(device), gts.to(device)
            
            optimizer.zero_grad()
            outputs = model(preds)
            loss = criterion(outputs, gts)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * preds.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for preds, gts in val_loader:
                preds, gts = preds.to(device), gts.to(device)
                outputs = model(preds)
                loss = criterion(outputs, gts)
                val_loss += loss.item() * preds.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1:03d}/{100} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = f"mlp_{coord_type}_refiner.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model to {save_path}")

def train_mlp(args):
    """Main function to train both MLP models."""
    print("="*80)
    print("🚀 Stage 2: Training MLP Refinement Models")
    print("="*80)
    
    if not os.path.exists(args.mlp_data_path):
        print(f"✗ ERROR: MLP data file not found at {args.mlp_data_path}")
        print("  Please run with --generate-data first.")
        return
        
    train_mlp_model(args, 'x')
    train_mlp_model(args, 'y')
    
    print("\n✅ Successfully trained both MLP refinement models.")

def main():
    parser = argparse.ArgumentParser(
        description="Train an MLP refinement stage for HRNetV2 landmark detection."
    )
    
    # --- Arguments for Data Generation ---
    gen_group = parser.add_argument_group('Data Generation')
    gen_group.add_argument('--generate-data', action='store_true', help='Run the data generation stage.')
    gen_group.add_argument('--hrnet-config', type=str, default='Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py', help='Path to the HRNetV2 config file.')
    gen_group.add_argument('--hrnet-checkpoint', type=str, default=None, help='Path to the HRNetV2 model checkpoint. If not given, finds the latest best model in work_dir.')
    gen_group.add_argument('--data-path', type=str, default="/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json", help='Path to the JSON file with image data and ground truth.')
    
    # --- Arguments for MLP Training ---
    train_group = parser.add_argument_group('MLP Training')
    train_group.add_argument('--train-mlp', action='store_true', help='Run the MLP training stage.')
    
    # --- Shared Arguments ---
    parser.add_argument('--mlp-data-path', type=str, default='mlp_training_data.csv', help='Path to save/load the intermediate MLP training data CSV.')

    args = parser.parse_args()

    if args.generate_data:
        generate_mlp_training_data(args)
    
    if args.train_mlp:
        train_mlp(args)
        
    if not args.generate_data and not args.train_mlp:
        print("Please specify a stage to run: --generate-data and/or --train-mlp")
        parser.print_help()

if __name__ == "__main__":
    # Apply PyTorch safe loading fix for MMPose checkpoints
    _original_torch_load = torch.load
    def safe_torch_load(*args, **kwargs):
        # MMPose checkpoints contain metadata (like HistoryBuffer) that requires
        # `weights_only=False`. Since we are using a self-trained, trusted model,
        # this is safe.
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = safe_torch_load

    main() 