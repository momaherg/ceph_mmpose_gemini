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
•   **Curriculum augmentation** – Start with light augments, gradually increase intensity
•   **Hard-example oversampling** – Oversample difficult samples with high error
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
        log_interval=20,
        hard_example_threshold=6.0,  # MRE threshold for hard examples
        curriculum_start_epoch=5,    # When to start curriculum augmentation
        max_oversample_ratio=2.0     # Maximum oversampling ratio for hard examples
    )
]
```

Make sure this file is importable (e.g. by placing it in PYTHONPATH or the
workspace root).
"""

from __future__ import annotations

import os
from typing import List, Dict, Tuple
import math
import random

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
    """In-memory dataset for joint 38-D MLP training with hard-example oversampling."""

    def __init__(self, preds: np.ndarray, gts: np.ndarray, sample_weights: np.ndarray = None):
        # preds/gts shape: [N, 38] (19 landmarks × 2 coordinates)
        assert preds.shape == gts.shape
        assert preds.shape[1] == 38
        self.preds = torch.from_numpy(preds).float()
        self.gts = torch.from_numpy(gts).float()
        
        # Handle sample weights for hard-example oversampling
        if sample_weights is not None:
            self.sample_weights = sample_weights
            # Create oversampled indices based on weights
            self.indices = self._create_oversampled_indices()
        else:
            self.sample_weights = np.ones(len(preds))
            self.indices = list(range(len(preds)))

    def _create_oversampled_indices(self):
        """Create indices with oversampling based on sample weights."""
        indices = []
        for i, weight in enumerate(self.sample_weights):
            # Add original sample
            indices.append(i)
            # Add additional copies based on weight
            additional_copies = int(weight - 1)
            for _ in range(additional_copies):
                indices.append(i)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.preds[actual_idx], self.gts[actual_idx]


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


class CurriculumAugmentation:
    """Curriculum learning for data augmentation intensity."""
    
    def __init__(self, start_epoch=5, max_epochs=100):
        self.start_epoch = start_epoch
        self.max_epochs = max_epochs
    
    def get_augmentation_params(self, current_epoch):
        """Get augmentation parameters based on current epoch."""
        if current_epoch < self.start_epoch:
            # No augmentation in early epochs
            return {
                'rotation_factor': 0,
                'scale_factor': (1.0, 1.0),
                'noise_std': 0.0,
                'shift_factor': 0.0
            }
        
        # Progressive augmentation intensity
        progress = min(1.0, (current_epoch - self.start_epoch) / (self.max_epochs - self.start_epoch))
        
        return {
            'rotation_factor': progress * 5.0,        # 0 -> 5 degrees
            'scale_factor': (1.0 - progress * 0.05, 1.0 + progress * 0.05),  # 0.95 -> 1.05
            'noise_std': progress * 0.5,             # 0 -> 0.5 pixel noise
            'shift_factor': progress * 0.02           # 0 -> 2% shift
        }


# -----------------------------------------------------------------------------
#  Hook implementation
# -----------------------------------------------------------------------------

@HOOKS.register_module()
class ConcurrentMLPTrainingHook(Hook):
    """MMEngine hook that performs concurrent joint MLP refinement training with curriculum learning."""

    priority = 'LOW'  # Run after default hooks

    def __init__(
        self,
        mlp_epochs: int = 100,
        mlp_batch_size: int = 16,
        mlp_lr: float = 3e-4,  # Higher initial LR for cosine schedule
        mlp_weight_decay: float = 1e-4,
        log_interval: int = 20,
        hard_example_threshold: float = 6.0,  # MRE threshold for hard examples
        curriculum_start_epoch: int = 5,      # When to start curriculum augmentation
        max_oversample_ratio: float = 2.0,    # Maximum oversampling ratio
    ) -> None:
        self.mlp_epochs = mlp_epochs
        self.mlp_batch_size = mlp_batch_size
        self.mlp_lr = mlp_lr
        self.mlp_weight_decay = mlp_weight_decay
        self.log_interval = log_interval
        self.hard_example_threshold = hard_example_threshold
        self.curriculum_start_epoch = curriculum_start_epoch
        self.max_oversample_ratio = max_oversample_ratio

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
        
        # Hard-example tracking
        self.sample_errors: Dict[int, float] = {}  # sample_id -> MRE
        self.hard_examples: List[int] = []         # list of hard example indices
        
        # Curriculum augmentation
        self.curriculum = CurriculumAugmentation(
            start_epoch=curriculum_start_epoch,
            max_epochs=mlp_epochs
        )

    # ---------------------------------------------------------------------
    # MMEngine lifecycle methods
    # ---------------------------------------------------------------------

    def before_run(self, runner: Runner):
        logger: MMLogger = runner.logger
        logger.info('[ConcurrentMLPTrainingHook] Initialising joint 38-D MLP model with curriculum learning…')

        # Initialize joint MLP model
        self.mlp_model = JointMLPRefinementModel(input_dim=38, hidden_dim=512, output_dim=38).to(self.device)
        self.optimizer = optim.Adam(self.mlp_model.parameters(), lr=self.mlp_lr, weight_decay=self.mlp_weight_decay)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.mlp_epochs, eta_min=1e-6)
        
        # Initialize landmark-wise scalers (19 for each coordinate type)
        self.landmark_scalers_input = [StandardScaler() for _ in range(19)]
        self.landmark_scalers_target = [StandardScaler() for _ in range(19)]
        
        logger.info(f'[ConcurrentMLPTrainingHook] Model architecture: 38 → 512 → 512 → 38 with residual connection')
        logger.info(f'[ConcurrentMLPTrainingHook] Parameters: {sum(p.numel() for p in self.mlp_model.parameters()):,}')
        logger.info(f'[ConcurrentMLPTrainingHook] Hard-example threshold: {self.hard_example_threshold} pixels')
        logger.info(f'[ConcurrentMLPTrainingHook] Curriculum starts at epoch: {self.curriculum_start_epoch}')
        logger.info(f'[ConcurrentMLPTrainingHook] Max oversample ratio: {self.max_oversample_ratio}x')

    def _compute_sample_errors(self, predictions: np.ndarray, ground_truths: np.ndarray) -> np.ndarray:
        """Compute per-sample MRE for hard-example identification."""
        # predictions, ground_truths shape: [N, 19, 2]
        errors = np.sqrt(np.sum((predictions - ground_truths)**2, axis=2))  # [N, 19]
        sample_mres = np.mean(errors, axis=1)  # [N]
        return sample_mres

    def _identify_hard_examples(self, sample_errors: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """Identify hard examples and compute sample weights for oversampling."""
        hard_indices = []
        sample_weights = np.ones(len(sample_errors))
        
        for i, error in enumerate(sample_errors):
            if error > self.hard_example_threshold:
                hard_indices.append(i)
                # Weight proportional to error, capped at max_oversample_ratio
                weight = min(self.max_oversample_ratio, error / self.hard_example_threshold)
                sample_weights[i] = weight
        
        return hard_indices, sample_weights

    def _apply_curriculum_augmentation(self, predictions: np.ndarray, ground_truths: np.ndarray, 
                                     current_epoch: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply curriculum-based data augmentation."""
        aug_params = self.curriculum.get_augmentation_params(current_epoch)
        
        if current_epoch < self.curriculum_start_epoch:
            # No augmentation in early epochs
            return predictions, ground_truths
        
        augmented_preds = []
        augmented_gts = []
        
        for pred, gt in zip(predictions, ground_truths):
            # Apply random transformations
            if random.random() < 0.3:  # 30% chance of augmentation
                # Small rotation (simulate head tilt variations)
                if aug_params['rotation_factor'] > 0:
                    angle = random.uniform(-aug_params['rotation_factor'], aug_params['rotation_factor'])
                    angle_rad = np.radians(angle)
                    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                    
                    # Center coordinates around image center for rotation
                    center = np.array([112, 112])  # 224x224 image center
                    pred_centered = pred - center
                    gt_centered = gt - center
                    
                    # Apply rotation matrix
                    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                    pred_rotated = pred_centered @ rotation_matrix.T + center
                    gt_rotated = gt_centered @ rotation_matrix.T + center
                    
                    pred = pred_rotated
                    gt = gt_rotated
                
                # Small scale variation
                scale_min, scale_max = aug_params['scale_factor']
                if scale_min != 1.0 or scale_max != 1.0:
                    scale = random.uniform(scale_min, scale_max)
                    center = np.array([112, 112])
                    pred = (pred - center) * scale + center
                    gt = (gt - center) * scale + center
                
                # Small coordinate noise
                if aug_params['noise_std'] > 0:
                    noise = np.random.normal(0, aug_params['noise_std'], pred.shape)
                    pred = pred + noise
                    # Don't add noise to ground truth
                
                # Small shift
                if aug_params['shift_factor'] > 0:
                    shift_x = random.uniform(-aug_params['shift_factor'] * 224, aug_params['shift_factor'] * 224)
                    shift_y = random.uniform(-aug_params['shift_factor'] * 224, aug_params['shift_factor'] * 224)
                    shift = np.array([shift_x, shift_y])
                    pred = pred + shift
                    gt = gt + shift
                
                # Ensure coordinates stay within image bounds
                pred = np.clip(pred, 0, 224)
                gt = np.clip(gt, 0, 224)
            
            augmented_preds.append(pred)
            augmented_gts.append(gt)
        
        return np.array(augmented_preds), np.array(augmented_gts)

    def after_train_epoch(self, runner: Runner):
        """After each HRNetV2 epoch, train joint MLP on-the-fly using current predictions."""
        logger: MMLogger = runner.logger
        assert self.mlp_model is not None

        current_hrnet_epoch = runner.epoch + 1
        logger.info(f'[ConcurrentMLPTrainingHook] Starting MLP training for HRNet epoch {current_hrnet_epoch}')

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
        sample_ids: List[int] = []        # Track sample IDs for hard-example identification

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
                            sample_ids.append(idx)
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
        # Step 1.5: Hard-example identification and curriculum augmentation
        # -----------------------------------------------------------------
        
        # Compute sample errors for hard-example identification
        sample_errors = self._compute_sample_errors(all_preds, all_gts)
        hard_indices, sample_weights = self._identify_hard_examples(sample_errors)
        
        logger.info(f'[ConcurrentMLPTrainingHook] Identified {len(hard_indices)} hard examples (>{self.hard_example_threshold:.1f}px MRE)')
        if hard_indices:
            avg_hard_error = np.mean([sample_errors[i] for i in hard_indices])
            max_weight = np.max(sample_weights)
            logger.info(f'[ConcurrentMLPTrainingHook] Hard examples avg MRE: {avg_hard_error:.2f}px, max weight: {max_weight:.2f}x')
        
        # Apply curriculum augmentation
        if current_hrnet_epoch >= self.curriculum_start_epoch:
            aug_params = self.curriculum.get_augmentation_params(current_hrnet_epoch)
            logger.info(f'[ConcurrentMLPTrainingHook] Curriculum augmentation active - rotation: ±{aug_params["rotation_factor"]:.1f}°, '
                       f'scale: {aug_params["scale_factor"][0]:.3f}-{aug_params["scale_factor"][1]:.3f}, '
                       f'noise: {aug_params["noise_std"]:.2f}px')
            
            all_preds, all_gts = self._apply_curriculum_augmentation(all_preds, all_gts, current_hrnet_epoch)
        else:
            logger.info(f'[ConcurrentMLPTrainingHook] Curriculum augmentation inactive (epoch {current_hrnet_epoch} < {self.curriculum_start_epoch})')

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
        # Step 3: Train joint MLP with hard-example oversampling
        # -----------------------------------------------------------------
        logger.info('[ConcurrentMLPTrainingHook] Training joint 38-D MLP with hard-example oversampling...')
        
        # Calculate initial loss for logging
        initial_loss = self.criterion(
            torch.from_numpy(preds_flat).float(),
            torch.from_numpy(gts_flat).float()
        ).item()
        
        # Create dataset and dataloader with sample weights for oversampling
        dataset = _JointMLPDataset(preds_flat, gts_flat, sample_weights)
        dataloader = data.DataLoader(dataset, batch_size=self.mlp_batch_size, shuffle=True, pin_memory=True)
        
        logger.info(f'[ConcurrentMLPTrainingHook] Dataset size after oversampling: {len(dataset)} (original: {len(preds_flat)})')
        
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
                           f'Loss: {avg_loss:.6f}, Initial: {initial_loss:.6f}, LR: {current_lr:.2e}, '
                           f'Hard examples: {len(hard_indices)}')

        logger.info('[ConcurrentMLPTrainingHook] Finished joint MLP training.')
        
        # Save models
        save_dir = os.path.join(runner.work_dir, 'concurrent_mlp')
        os.makedirs(save_dir, exist_ok=True)
        
        current_epoch = runner.epoch + 1
        model_epoch_path = os.path.join(save_dir, f'joint_mlp_epoch_{current_epoch}.pth')
        model_latest_path = os.path.join(save_dir, 'joint_mlp_latest.pth')
        
        torch.save(self.mlp_model.state_dict(), model_epoch_path)
        torch.save(self.mlp_model.state_dict(), model_latest_path)
        
        # Save hard-example statistics
        hard_example_stats = {
            'epoch': current_epoch,
            'total_samples': len(all_preds),
            'hard_examples': len(hard_indices),
            'hard_example_ratio': len(hard_indices) / len(all_preds),
            'avg_sample_error': float(np.mean(sample_errors)),
            'hard_example_threshold': self.hard_example_threshold,
            'curriculum_active': current_hrnet_epoch >= self.curriculum_start_epoch
        }
        
        import json
        stats_path = os.path.join(save_dir, f'hard_example_stats_epoch_{current_epoch}.json')
        with open(stats_path, 'w') as f:
            json.dump(hard_example_stats, f, indent=2)
        
        logger.info(f'[ConcurrentMLPTrainingHook] Joint MLP saved: {model_latest_path}')
        logger.info(f'[ConcurrentMLPTrainingHook] Hard-example stats saved: {stats_path}')

    def after_run(self, runner: Runner):
        logger: MMLogger = runner.logger
        if self.mlp_model is None:
            return
        save_dir = os.path.join(runner.work_dir, 'concurrent_mlp')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.mlp_model.state_dict(), os.path.join(save_dir, 'joint_mlp_final.pth'))
        logger.info(f'[ConcurrentMLPTrainingHook] Saved final joint MLP weights to {save_dir}') 