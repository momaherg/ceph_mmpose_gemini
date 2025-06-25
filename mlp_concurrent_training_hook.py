#!/usr/bin/env python3
"""
Concurrent MLP Training Hook for MMEngine / MMPose - SIMPLIFIED VERSION
-----------------------------------------------------------------
This hook trains two MLP refinement models (one for X, one for Y) **concurrently**
with HRNetV2 training. After every HRNet training epoch, the hook:

1. Uses the training dataloader directly to get model predictions
2. Collects predicted and ground truth coordinates 
3. Trains each MLP for 100 epochs

Key improvements in this version:
• **Simplified data access** – Uses the training dataloader directly
• **Robust inference** – Uses model.test_step() which handles all metadata
• **No complex setup** – Avoids all the metadata issues
• **GPU optimized** – All operations on GPU
"""

from __future__ import annotations

import os
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.logging import MMLogger
from mmengine.runner import Runner

import numpy as np

# -----------------------------------------------------------------------------
#  MLP architecture
# -----------------------------------------------------------------------------

class MLPRefinementModel(nn.Module):
    """Simple 19→500→19 fully connected network with ReLU + dropout."""

    def __init__(self, input_dim: int = 19, hidden_dim: int = 500, output_dim: int = 19):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class _MLPDataset(data.Dataset):
    """In-memory dataset of predicted → ground-truth coordinates."""

    def __init__(self, preds: np.ndarray, gts: np.ndarray):
        # preds/gts shape: [N, 19]
        assert preds.shape == gts.shape
        self.preds = torch.from_numpy(preds).float()
        self.gts = torch.from_numpy(gts).float()

    def __len__(self):
        return self.preds.shape[0]

    def __getitem__(self, idx):
        return self.preds[idx], self.gts[idx]


# -----------------------------------------------------------------------------
#  Hook implementation - SIMPLIFIED
# -----------------------------------------------------------------------------

@HOOKS.register_module()
class ConcurrentMLPTrainingHook(Hook):
    """MMEngine hook that performs concurrent MLP refinement training - SIMPLIFIED."""

    priority = 'LOW'  # Run after default hooks

    def __init__(
        self,
        mlp_epochs: int = 100,
        mlp_batch_size: int = 16,
        mlp_lr: float = 1e-5,
        mlp_weight_decay: float = 1e-4,
        log_interval: int = 50,
    ) -> None:
        self.mlp_epochs = mlp_epochs
        self.mlp_batch_size = mlp_batch_size
        self.mlp_lr = mlp_lr
        self.mlp_weight_decay = mlp_weight_decay
        self.log_interval = log_interval

        # These will be initialised in before_run
        self.mlp_x: MLPRefinementModel | None = None
        self.mlp_y: MLPRefinementModel | None = None
        self.opt_x: optim.Optimizer | None = None
        self.opt_y: optim.Optimizer | None = None
        self.criterion = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def before_run(self, runner: Runner):
        logger: MMLogger = runner.logger
        logger.info('[ConcurrentMLPTrainingHook] Initialising MLP models …')

        self.mlp_x = MLPRefinementModel().to(self.device)
        self.mlp_y = MLPRefinementModel().to(self.device)
        self.opt_x = optim.Adam(self.mlp_x.parameters(), lr=self.mlp_lr, weight_decay=self.mlp_weight_decay)
        self.opt_y = optim.Adam(self.mlp_y.parameters(), lr=self.mlp_lr, weight_decay=self.mlp_weight_decay)

    def after_train_epoch(self, runner: Runner):
        """After each HRNetV2 epoch, train MLP on-the-fly using current predictions."""
        logger: MMLogger = runner.logger
        assert self.mlp_x is not None and self.mlp_y is not None

        logger.info('[ConcurrentMLPTrainingHook] Generating predictions using training dataloader...')

        # -----------------------------------------------------------------
        # Step 1: Use training dataloader to get predictions (SIMPLIFIED)
        # -----------------------------------------------------------------
        
        model = runner.model
        model.eval()
        
        preds_x: List[np.ndarray] = []
        preds_y: List[np.ndarray] = []
        gts_x: List[np.ndarray] = []
        gts_y: List[np.ndarray] = []

        # Helper function to safely convert tensor to numpy
        def tensor_to_numpy(data):
            if isinstance(data, torch.Tensor):
                return data.cpu().numpy()
            elif isinstance(data, np.ndarray):
                return data
            else:
                return np.array(data)

        # Use the training dataloader directly - this handles all the complex preprocessing
        train_dataloader = runner.train_dataloader
        processed_samples = 0
        
        with torch.no_grad():
            for batch_idx, data_batch in enumerate(train_dataloader):
                try:
                    # data_batch contains properly preprocessed inputs and data_samples
                    inputs = data_batch['inputs']  # Already preprocessed images
                    data_samples = data_batch['data_samples']  # Contains ground truth
                    
                    # Move inputs to device
                    inputs = inputs.to(self.device)
                    
                    # Run model test_step (this handles all the metadata automatically)
                    predictions = model.test_step(data_batch)
                    
                    # Extract predictions and ground truth from batch
                    batch_size = len(data_samples)
                    
                    for i in range(batch_size):
                        try:
                            # Get predicted keypoints
                            if hasattr(predictions[i], 'pred_instances') and hasattr(predictions[i].pred_instances, 'keypoints'):
                                pred_kpts = tensor_to_numpy(predictions[i].pred_instances.keypoints[0])
                                
                                # Get ground truth keypoints
                                if hasattr(data_samples[i], 'gt_instances') and hasattr(data_samples[i].gt_instances, 'keypoints'):
                                    gt_kpts = tensor_to_numpy(data_samples[i].gt_instances.keypoints[0])
                                    
                                    # Validate shapes
                                    if pred_kpts.shape == (19, 2) and gt_kpts.shape == (19, 2):
                                        # Store coordinates
                                        preds_x.append(pred_kpts[:, 0])
                                        preds_y.append(pred_kpts[:, 1])
                                        gts_x.append(gt_kpts[:, 0])
                                        gts_y.append(gt_kpts[:, 1])
                                        processed_samples += 1
                                        
                        except Exception as e:
                            # Skip problematic samples silently
                            continue
                            
                    # Log progress periodically
                    if (batch_idx + 1) % 10 == 0:
                        logger.info(f'[ConcurrentMLPTrainingHook] Processed {batch_idx + 1}/{len(train_dataloader)} batches, {processed_samples} valid samples')
                        
                except Exception as e:
                    logger.warning(f'[ConcurrentMLPTrainingHook] Failed to process batch {batch_idx}: {e}')
                    continue

        logger.info(f'[ConcurrentMLPTrainingHook] Successfully processed {processed_samples} samples')

        if processed_samples == 0:
            logger.warning('[ConcurrentMLPTrainingHook] No valid predictions generated; skipping MLP update.')
            return

        # Convert lists to numpy arrays
        preds_x = np.stack(preds_x)
        preds_y = np.stack(preds_y)
        gts_x = np.stack(gts_x)
        gts_y = np.stack(gts_y)

        # -----------------------------------------------------------------
        # Step 2: Train MLPs (same as before)
        # -----------------------------------------------------------------
        logger.info('[ConcurrentMLPTrainingHook] Training MLPs on GPU…')

        # Build datasets and loaders
        ds_x = _MLPDataset(preds_x, gts_x)
        ds_y = _MLPDataset(preds_y, gts_y)
        dl_x = data.DataLoader(ds_x, batch_size=self.mlp_batch_size, shuffle=True, pin_memory=True)
        dl_y = data.DataLoader(ds_y, batch_size=self.mlp_batch_size, shuffle=True, pin_memory=True)

        def _train_one(model: MLPRefinementModel, optimiser: optim.Optimizer, loader: data.DataLoader, name: str):
            model.train()
            for ep in range(self.mlp_epochs):
                epoch_loss = 0.0
                for preds_batch, gts_batch in loader:
                    preds_batch = preds_batch.to(self.device, non_blocking=True)
                    gts_batch = gts_batch.to(self.device, non_blocking=True)

                    optimiser.zero_grad()
                    outputs = model(preds_batch)
                    loss = self.criterion(outputs, gts_batch)
                    loss.backward()
                    optimiser.step()

                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(loader)
                if (ep + 1) % 20 == 0:
                    logger.info(f'[ConcurrentMLPTrainingHook] {name} epoch {ep+1}/{self.mlp_epochs} loss: {avg_loss:.6f}')

        _train_one(self.mlp_x, self.opt_x, dl_x, 'MLP-X')
        _train_one(self.mlp_y, self.opt_y, dl_y, 'MLP-Y')

        logger.info('[ConcurrentMLPTrainingHook] Finished MLP update for this HRNet epoch.')

    def after_run(self, runner: Runner):
        logger: MMLogger = runner.logger
        if self.mlp_x is None or self.mlp_y is None:
            return
        save_dir = os.path.join(runner.work_dir, 'concurrent_mlp')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.mlp_x.state_dict(), os.path.join(save_dir, 'mlp_x_final.pth'))
        torch.save(self.mlp_y.state_dict(), os.path.join(save_dir, 'mlp_y_final.pth'))
        logger.info(f'[ConcurrentMLPTrainingHook] Saved final MLP weights to {save_dir}') 