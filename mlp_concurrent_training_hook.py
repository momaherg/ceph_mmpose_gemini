#!/usr/bin/env python3
"""
Concurrent MLP Training Hook for MMEngine / MMPose
-------------------------------------------------
This hook trains a joint MLP refinement model **concurrently** with HRNetV2 training.
After every HRNet training epoch, the hook:

1.  Runs inference on the entire *training* dataloader using the *current*
    HRNetV2 weights to obtain predicted landmark coordinates.
2.  Creates an in-memory dataset of (predicted → ground-truth) coordinate pairs.
3.  Trains a joint 38-D MLP for a fixed number of epochs (default: 100).

Key improvements:
•   **Landmark-wise normalization** – Each landmark gets its own scaler for better handling of different coordinate ranges
•   **Joint 38-D MLP** – Single model learns X,Y correlations instead of separate models
•   **Cosine LR scheduling** – Better convergence than fixed learning rate
•   **One-time initialisation** – MLP weights, optimisers and scalers are created once and persist
•   **No gradient leakage** – MLP training is completely detached from HRNetV2

To enable this hook, add to your config:

```
custom_hooks = [
    dict(
        type='ConcurrentMLPTrainingHook',
        mlp_epochs=100,
        mlp_batch_size=16,
        mlp_lr=3e-4,  # Higher initial LR for cosine schedule
        mlp_weight_decay=1e-4,
        log_interval=20
    )
]
```

Make sure this file is importable (e.g. by placing it in PYTHONPATH or the
workspace root).
"""

from __future__ import annotations

import os
from typing import List
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.logging import MMLogger
from mmengine.runner import Runner

from mmpose.apis import inference_topdown
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# -----------------------------------------------------------------------------
#  Joint 38-D MLP architecture with residual connections
# -----------------------------------------------------------------------------

class JointMLPRefinementModel(nn.Module):
    """
    Joint MLP model for landmark coordinate refinement.
    Input: 38 predicted coordinates (19 landmarks × 2 coordinates)
    Hidden: 512 neurons with residual connection
    Output: 38 refined coordinates
    """
    def __init__(self, input_dim=38, hidden_dim=512, output_dim=38):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Main network with residual connection
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Residual connection (input -> output)
        self.residual = nn.Linear(input_dim, output_dim)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Main path
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        # Residual connection
        residual = self.residual(x)
        
        # Combine main path and residual
        out = out + residual
        
        return out


class _JointMLPDataset(data.Dataset):
    """In-memory dataset for joint 38-D MLP training."""

    def __init__(self, preds: np.ndarray, gts: np.ndarray):
        # preds/gts shape: [N, 38] (19 landmarks × 2 coordinates)
        assert preds.shape == gts.shape
        assert preds.shape[1] == 38
        self.preds = torch.from_numpy(preds).float()
        self.gts = torch.from_numpy(gts).float()

    def __len__(self):
        return self.preds.shape[0]

    def __getitem__(self, idx):
        return self.preds[idx], self.gts[idx]


class CosineAnnealingLR:
    """Simple cosine annealing learning rate scheduler."""
    
    def __init__(self, optimizer, T_max, eta_min=1e-6):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch):
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2


# -----------------------------------------------------------------------------
#  Hook implementation
# -----------------------------------------------------------------------------

@HOOKS.register_module()
class ConcurrentMLPTrainingHook(Hook):
    """MMEngine hook that performs concurrent joint MLP refinement training."""

    priority = 'LOW'  # Run after default hooks

    def __init__(
        self,
        mlp_epochs: int = 100,
        mlp_batch_size: int = 16,
        mlp_lr: float = 3e-4,  # Higher initial LR for cosine schedule
        mlp_weight_decay: float = 1e-4,
        log_interval: int = 20,
    ) -> None:
        self.mlp_epochs = mlp_epochs
        self.mlp_batch_size = mlp_batch_size
        self.mlp_lr = mlp_lr
        self.mlp_weight_decay = mlp_weight_decay
        self.log_interval = log_interval

        # These will be initialised in before_run
        self.mlp_model: JointMLPRefinementModel | None = None
        self.optimizer: optim.Optimizer | None = None
        self.lr_scheduler: CosineAnnealingLR | None = None
        self.criterion = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Landmark-wise normalization scalers (19 scalers each for input and target)
        self.landmark_scalers_input: List[StandardScaler] = []
        self.landmark_scalers_target: List[StandardScaler] = []
        self.scalers_initialized = False

    # ---------------------------------------------------------------------
    # MMEngine lifecycle methods
    # ---------------------------------------------------------------------

    def before_run(self, runner: Runner):
        logger: MMLogger = runner.logger
        logger.info('[ConcurrentMLPTrainingHook] Initialising joint 38-D MLP model with landmark-wise scalers…')

        # Initialize joint MLP model
        self.mlp_model = JointMLPRefinementModel(input_dim=38, hidden_dim=512, output_dim=38).to(self.device)
        self.optimizer = optim.Adam(self.mlp_model.parameters(), lr=self.mlp_lr, weight_decay=self.mlp_weight_decay)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.mlp_epochs, eta_min=1e-6)
        
        # Initialize landmark-wise scalers (19 for each coordinate type)
        self.landmark_scalers_input = [StandardScaler() for _ in range(19)]
        self.landmark_scalers_target = [StandardScaler() for _ in range(19)]
        
        logger.info(f'[ConcurrentMLPTrainingHook] Model architecture: 38 → 512 → 512 → 38 with residual connection')
        logger.info(f'[ConcurrentMLPTrainingHook] Parameters: {sum(p.numel() for p in self.mlp_model.parameters()):,}')

    def after_train_epoch(self, runner: Runner):
        """After each HRNetV2 epoch, train joint MLP on-the-fly using current predictions."""
        logger: MMLogger = runner.logger
        assert self.mlp_model is not None

        # -----------------------------------------------------------------
        # Step 1: Generate predictions on training data
        # -----------------------------------------------------------------
        
        # Get the actual model, handling potential wrapping
        model = runner.model
        if hasattr(model, 'module'):
            actual_model = model.module
        else:
            actual_model = model
            
        # Ensure model has required attributes for inference
        if not hasattr(actual_model, 'cfg') and hasattr(runner, 'cfg'):
            actual_model.cfg = runner.cfg
            
        # Set up dataset_meta if missing
        if not hasattr(actual_model, 'dataset_meta'):
            try:
                import cephalometric_dataset_info
                dataset_meta = {
                    'dataset_name': 'cephalometric',
                    'joint_weights': cephalometric_dataset_info.dataset_info.get('joint_weights', [1.0] * 19),
                    'sigmas': cephalometric_dataset_info.dataset_info.get('sigmas', [0.035] * 19),
                    'flip_indices': cephalometric_dataset_info.dataset_info.get('flip_indices', list(range(19))),
                    'keypoint_info': cephalometric_dataset_info.dataset_info.get('keypoint_info', {}),
                    'skeleton_info': cephalometric_dataset_info.dataset_info.get('skeleton_info', []),
                    'keypoint_name2id': {f'keypoint_{i}': i for i in range(19)},
                    'keypoint_id2name': {i: f'keypoint_{i}' for i in range(19)},
                }
                actual_model.dataset_meta = dataset_meta
                logger.info('[ConcurrentMLPTrainingHook] Set dataset_meta for inference compatibility')
            except Exception as e:
                logger.warning(f'[ConcurrentMLPTrainingHook] Could not set dataset_meta: {e}')
        
        model.eval()
        all_preds: List[np.ndarray] = []  # Will store [N, 19, 2] predictions
        all_gts: List[np.ndarray] = []    # Will store [N, 19, 2] ground truths

        train_dataset = runner.train_dataloader.dataset
        logger.info('[ConcurrentMLPTrainingHook] Generating predictions for joint MLP...')

        def tensor_to_numpy(data):
            """Safely convert tensor to numpy."""
            if isinstance(data, torch.Tensor):
                return data.cpu().numpy()
            elif isinstance(data, np.ndarray):
                return data
            else:
                return np.array(data)

        # Generate training data
        try:
            from tqdm import tqdm
            
            if hasattr(train_dataset, 'ann_file') and train_dataset.ann_file:
                logger.info(f'[ConcurrentMLPTrainingHook] Loading data from: {train_dataset.ann_file}')
                
                try:
                    df = pd.read_json(train_dataset.ann_file)
                    logger.info(f'[ConcurrentMLPTrainingHook] Loaded {len(df)} samples')
                    
                    import cephalometric_dataset_info
                    landmark_names = cephalometric_dataset_info.landmark_names_in_order
                    landmark_cols = cephalometric_dataset_info.original_landmark_cols
                    
                    processed_count = 0
                    
                    for idx, row in tqdm(df.iterrows(), total=len(df), disable=runner.rank != 0, desc="Generating joint MLP data"):
                        try:
                            # Get image
                            img_array = np.array(row['Image'], dtype=np.uint8).reshape((224, 224, 3))
                            
                            # Get ground truth keypoints
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
                                
                            gt_keypoints = np.array(gt_keypoints)  # Shape: [19, 2]
                            
                            # Run inference
                            bbox = np.array([[0, 0, 224, 224]], dtype=np.float32)
                            results = inference_topdown(model, img_array, bboxes=bbox, bbox_format='xyxy')
                            
                            if results and len(results) > 0:
                                pred_keypoints = results[0].pred_instances.keypoints[0]
                                pred_keypoints = tensor_to_numpy(pred_keypoints)  # Shape: [19, 2]
                            else:
                                continue
                            
                            if pred_keypoints is None or pred_keypoints.shape[0] != 19:
                                continue
                            
                            # Store predictions and ground truths
                            all_preds.append(pred_keypoints)
                            all_gts.append(gt_keypoints)
                            processed_count += 1
                            
                        except Exception as e:
                            if processed_count < 5:  # Only log first few errors
                                logger.warning(f'[ConcurrentMLPTrainingHook] Failed to process sample {idx}: {e}')
                            continue
                    
                    logger.info(f'[ConcurrentMLPTrainingHook] Successfully processed {processed_count} samples')
                    
                except Exception as e:
                    logger.error(f'[ConcurrentMLPTrainingHook] Failed to load annotation file: {e}')
                    return

        except Exception as e:
            logger.error(f'[ConcurrentMLPTrainingHook] Critical error during data processing: {e}')
            return

        if not all_preds:
            logger.warning('[ConcurrentMLPTrainingHook] No predictions generated; skipping MLP update.')
            return

        # Convert to numpy arrays
        all_preds = np.array(all_preds)  # Shape: [N, 19, 2]
        all_gts = np.array(all_gts)      # Shape: [N, 19, 2]
        
        logger.info(f'[ConcurrentMLPTrainingHook] Generated data: {all_preds.shape[0]} samples, {all_preds.shape[1]} landmarks')

        # -----------------------------------------------------------------
        # Step 2: Landmark-wise normalization
        # -----------------------------------------------------------------
        
        if not self.scalers_initialized:
            logger.info('[ConcurrentMLPTrainingHook] Initializing landmark-wise scalers...')
            
            # Fit scalers for each landmark separately
            for landmark_idx in range(19):
                # Input scalers (predictions)
                pred_coords = all_preds[:, landmark_idx, :].reshape(-1, 2)  # [N, 2]
                self.landmark_scalers_input[landmark_idx].fit(pred_coords)
                
                # Target scalers (ground truth)
                gt_coords = all_gts[:, landmark_idx, :].reshape(-1, 2)  # [N, 2]
                self.landmark_scalers_target[landmark_idx].fit(gt_coords)
            
            self.scalers_initialized = True
            logger.info('[ConcurrentMLPTrainingHook] Landmark-wise scalers initialized')
            
            # Save scalers
            save_dir = os.path.join(runner.work_dir, 'concurrent_mlp')
            os.makedirs(save_dir, exist_ok=True)
            
            joblib.dump(self.landmark_scalers_input, os.path.join(save_dir, 'landmark_scalers_input.pkl'))
            joblib.dump(self.landmark_scalers_target, os.path.join(save_dir, 'landmark_scalers_target.pkl'))
            
            logger.info(f'[ConcurrentMLPTrainingHook] Landmark-wise scalers saved to {save_dir}')

        # Apply landmark-wise normalization
        preds_normalized = np.zeros_like(all_preds)
        gts_normalized = np.zeros_like(all_gts)
        
        for landmark_idx in range(19):
            # Normalize predictions
            pred_coords = all_preds[:, landmark_idx, :].reshape(-1, 2)
            pred_coords_norm = self.landmark_scalers_input[landmark_idx].transform(pred_coords)
            preds_normalized[:, landmark_idx, :] = pred_coords_norm.reshape(-1, 2)
            
            # Normalize ground truth
            gt_coords = all_gts[:, landmark_idx, :].reshape(-1, 2)
            gt_coords_norm = self.landmark_scalers_target[landmark_idx].transform(gt_coords)
            gts_normalized[:, landmark_idx, :] = gt_coords_norm.reshape(-1, 2)

        # Flatten to 38-D vectors for joint MLP
        preds_flat = preds_normalized.reshape(-1, 38)  # [N, 38]
        gts_flat = gts_normalized.reshape(-1, 38)      # [N, 38]

        # -----------------------------------------------------------------
        # Step 3: Train joint MLP
        # -----------------------------------------------------------------
        logger.info('[ConcurrentMLPTrainingHook] Training joint 38-D MLP...')
        
        # Calculate initial loss for logging
        initial_loss = self.criterion(
            torch.from_numpy(preds_flat).float(),
            torch.from_numpy(gts_flat).float()
        ).item()
        
        # Create dataset and dataloader
        dataset = _JointMLPDataset(preds_flat, gts_flat)
        dataloader = data.DataLoader(dataset, batch_size=self.mlp_batch_size, shuffle=True, pin_memory=True)
        
        # Training loop
        self.mlp_model.train()
        for epoch in range(self.mlp_epochs):
            epoch_loss = 0.0
            
            for batch_preds, batch_gts in dataloader:
                batch_preds = batch_preds.to(self.device, non_blocking=True)
                batch_gts = batch_gts.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad()
                outputs = self.mlp_model(batch_preds)
                loss = self.criterion(outputs, batch_gts)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Update learning rate
            self.lr_scheduler.step(epoch)
            
            avg_loss = epoch_loss / len(dataloader)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            if (epoch + 1) % self.log_interval == 0:
                logger.info(f'[ConcurrentMLPTrainingHook] Joint MLP epoch {epoch+1}/{self.mlp_epochs} | '
                           f'Loss: {avg_loss:.6f}, Initial: {initial_loss:.6f}, LR: {current_lr:.2e}')

        logger.info('[ConcurrentMLPTrainingHook] Finished joint MLP training.')
        
        # Save models
        save_dir = os.path.join(runner.work_dir, 'concurrent_mlp')
        os.makedirs(save_dir, exist_ok=True)
        
        current_epoch = runner.epoch + 1
        model_epoch_path = os.path.join(save_dir, f'joint_mlp_epoch_{current_epoch}.pth')
        model_latest_path = os.path.join(save_dir, 'joint_mlp_latest.pth')
        
        torch.save(self.mlp_model.state_dict(), model_epoch_path)
        torch.save(self.mlp_model.state_dict(), model_latest_path)
        
        logger.info(f'[ConcurrentMLPTrainingHook] Joint MLP saved: {model_latest_path}')

    def after_run(self, runner: Runner):
        logger: MMLogger = runner.logger
        if self.mlp_model is None:
            return
        save_dir = os.path.join(runner.work_dir, 'concurrent_mlp')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.mlp_model.state_dict(), os.path.join(save_dir, 'joint_mlp_final.pth'))
        logger.info(f'[ConcurrentMLPTrainingHook] Saved final joint MLP weights to {save_dir}') 