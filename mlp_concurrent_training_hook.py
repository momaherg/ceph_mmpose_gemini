#!/usr/bin/env python3
"""
Concurrent MLP Training Hook for MMEngine / MMPose with Checkpoint Synchronization
----------------------------------------------------------------------------------
This hook trains a joint MLP refinement model **concurrently** with HRNetV2 training.  
After every HRNet training epoch, the hook:

1.  Runs inference on the entire *training* dataloader using the *current*
    HRNetV2 weights to obtain predicted landmark coordinates.
2.  Creates an in-memory dataset of (predicted → ground-truth) coordinate pairs.
3.  Trains a joint MLP for a fixed number of epochs (default: 100).
4.  Implements hard-example oversampling for samples with high landmark errors.
5.  **NEW**: Saves synchronized MLP models whenever HRNet checkpoints are saved.
6.  **NEW**: Creates weighted samplers for next HRNet epoch to oversample hard examples.
7.  **NEW**: Also trains for patient classification (Class I, II, III).

Important design decisions:
•   **Joint 38-D model** – Single MLP that predicts all 38 coordinates (19 x,y pairs)
    allowing the network to learn cross-correlations between X and Y axes.
•   **Multi-task learning** – MLP also predicts patient classification (3 classes).
•   **Hard-example oversampling** – Samples with any landmark MRE > threshold get
    duplicated in both MLP training and next HRNet epoch for focused learning.
•   **Checkpoint synchronization** – MLP models are saved in sync with HRNet checkpoints
    ensuring perfect correspondence for evaluation.
•   **One-time initialisation** – MLP weights, optimisers and scalers are created
    once in `before_run` and *persist* across the whole HRNet training.
•   **No gradient leakage** – MLP training is completely detached from the
    HRNetV2 computation graph (`torch.no_grad()`), so gradients do **not**
    propagate back.
•   **CPU/GPU awareness** – Trains on GPU if available, else CPU.

To enable this hook, add to your config:

```
custom_hooks = [
    dict(
        type='ConcurrentMLPTrainingHook',
        mlp_epochs=100,
        mlp_batch_size=16,
        mlp_lr=1e-5,
        mlp_weight_decay=1e-4,
        hard_example_threshold=5.0,  # MRE threshold for oversampling
        hrnet_hard_example_weight=2.0,  # Weight for hard examples in HRNet training
        log_interval=20,
        classification_loss_weight=1.0,  # Weight for classification loss
        use_landmark_features_for_classification=True  # Use landmarks for classification
    )
]
```

Make sure this file is importable (e.g. by placing it in PYTHONPATH or the
workspace root).
"""

from __future__ import annotations

import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Import for weighted sampling
from torch.utils.data import WeightedRandomSampler

# -----------------------------------------------------------------------------
#  Joint MLP architecture for 38-D coordinate prediction + classification
# -----------------------------------------------------------------------------

class JointMLPRefinementModel(nn.Module):
    """Joint MLP model for landmark coordinate refinement with adaptive selection and classification.
    
    Input: 38 predicted coordinates (19 landmarks × 2 coordinates)
    Hidden: 500 neurons with residual connection
    Output: 
        - 38 refined coordinates with adaptive gating
        - 3 classification logits (Class I, II, III)
    
    This model learns:
    1. MLP refinements for coordinates
    2. Per-coordinate selection weights to choose between HRNet and MLP predictions
    3. Patient classification based on landmarks
    """

    def __init__(self, input_dim: int = 38, hidden_dim: int = 500, output_dim: int = 38, 
                 num_classes: int = 3, use_landmark_features_for_classification: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.use_landmark_features_for_classification = use_landmark_features_for_classification
        
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
        
        # Classification network - predicts patient class from landmarks
        self.classification_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
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
            Tuple of:
                - Adaptively selected coordinates [batch_size, 38]
                - Classification logits [batch_size, 3]
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
        
        # Get classification predictions
        class_logits = self.classification_net(x)
        
        return adaptive_output, class_logits


class _MLPDataset(data.Dataset):
    """In-memory dataset with hard-example oversampling and classification labels."""

    def __init__(self, preds: np.ndarray, gts: np.ndarray, gt_classes: np.ndarray, sample_weights: np.ndarray = None):
        # preds/gts shape: [N, 38] (flattened coordinates)
        # gt_classes shape: [N] (class labels 0, 1, 2)
        assert preds.shape == gts.shape
        assert len(preds) == len(gt_classes)
        
        self.preds = torch.from_numpy(preds).float()
        self.gts = torch.from_numpy(gts).float()
        self.gt_classes = torch.from_numpy(gt_classes).long()
        
        # Create weighted sampling indices for hard examples
        if sample_weights is not None:
            # Oversample hard examples by duplicating their indices
            self.indices = []
            for i, weight in enumerate(sample_weights):
                # Add base sample
                self.indices.append(i)
                # Add extra copies for hard examples (weight > 1)
                extra_copies = int(weight) - 1
                for _ in range(extra_copies):
                    self.indices.append(i)
            self.indices = np.array(self.indices)
        else:
            self.indices = np.arange(len(self.preds))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.preds[actual_idx], self.gts[actual_idx], self.gt_classes[actual_idx]


# -----------------------------------------------------------------------------
#  Hook implementation
# -----------------------------------------------------------------------------

@HOOKS.register_module()
class ConcurrentMLPTrainingHook(Hook):
    """MMEngine hook that performs concurrent joint MLP refinement training with checkpoint synchronization and classification."""

    priority = 'VERY_LOW'  # Run after checkpoint hooks to ensure synchronization

    def __init__(
        self,
        mlp_epochs: int = 100,
        mlp_batch_size: int = 16,
        mlp_lr: float = 1e-5,
        mlp_weight_decay: float = 1e-4,
        hard_example_threshold: float = 5.0,  # MRE threshold in pixels
        log_interval: int = 50,
        hrnet_hard_example_weight: float = 2.0,  # Weight multiplier for hard examples in HRNet training
        classification_loss_weight: float = 1.0,  # Weight for classification loss
        use_landmark_features_for_classification: bool = True,  # Use landmarks for classification
    ) -> None:
        self.mlp_epochs = mlp_epochs
        self.mlp_batch_size = mlp_batch_size
        self.mlp_lr = mlp_lr
        self.mlp_weight_decay = mlp_weight_decay
        self.hard_example_threshold = hard_example_threshold
        self.log_interval = log_interval
        self.hrnet_hard_example_weight = hrnet_hard_example_weight
        self.classification_loss_weight = classification_loss_weight
        self.use_landmark_features_for_classification = use_landmark_features_for_classification

        # These will be initialised in before_run
        self.mlp_joint: JointMLPRefinementModel | None = None
        self.opt_joint: optim.Optimizer | None = None
        self.criterion_regression = nn.MSELoss()
        self.criterion_classification = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Normalization scalers - initialized once and reused
        self.scaler_input: StandardScaler | None = None
        self.scaler_target: StandardScaler | None = None
        self.scalers_initialized = False
        
        # Store sample weights for HRNet training
        self.sample_weights_for_hrnet: np.ndarray | None = None
        self.sample_indices_mapping: List[int] | None = None
        
        # Track checkpoint synchronization
        self.checkpoint_mlp_mapping: dict = {}  # Maps HRNet checkpoint names to MLP model paths
        self.last_saved_checkpoint: str | None = None

    # ---------------------------------------------------------------------
    # MMEngine lifecycle methods
    # ---------------------------------------------------------------------

    def before_run(self, runner: Runner):
        logger: MMLogger = runner.logger
        logger.info('[ConcurrentMLPTrainingHook] Initialising joint 38-D MLP model with adaptive selection and classification…')

        self.mlp_joint = JointMLPRefinementModel(
            use_landmark_features_for_classification=self.use_landmark_features_for_classification
        ).to(self.device)
        self.opt_joint = optim.Adam(self.mlp_joint.parameters(), lr=self.mlp_lr, weight_decay=self.mlp_weight_decay)
        
        # Initialize scalers for 38-D input/output
        self.scaler_input = StandardScaler()
        self.scaler_target = StandardScaler()
        
        logger.info(f'[ConcurrentMLPTrainingHook] Joint MLP initialized with {sum(p.numel() for p in self.mlp_joint.parameters()):,} parameters')
        logger.info(f'[ConcurrentMLPTrainingHook] Architecture includes:')
        logger.info(f'[ConcurrentMLPTrainingHook]   - Refinement network: Predicts MLP-refined coordinates')
        logger.info(f'[ConcurrentMLPTrainingHook]   - Selection network: Learns to choose between HRNet and MLP per coordinate')
        logger.info(f'[ConcurrentMLPTrainingHook]   - Classification network: Predicts patient class (I, II, III)')
        logger.info(f'[ConcurrentMLPTrainingHook] Hard-example threshold: {self.hard_example_threshold} pixels')
        logger.info(f'[ConcurrentMLPTrainingHook] HRNet hard-example weight: {self.hrnet_hard_example_weight}x')
        logger.info(f'[ConcurrentMLPTrainingHook] Classification loss weight: {self.classification_loss_weight}')

    def _create_weighted_sampler_for_hrnet(self, runner: Runner, sample_weights: np.ndarray):
        """Create a weighted sampler for HRNetV2 training based on hard examples."""
        logger: MMLogger = runner.logger
        
        try:
            # Store weights and mapping for the next epoch
            self.sample_weights_for_hrnet = sample_weights.copy()
            self.sample_indices_mapping = list(range(len(sample_weights)))
            
            # Create weighted sampler
            # Convert numpy weights to torch tensor
            weights_tensor = torch.from_numpy(sample_weights).float()
            
            # Create weighted random sampler
            weighted_sampler = WeightedRandomSampler(
                weights=weights_tensor,
                num_samples=len(sample_weights),
                replacement=True  # Allow replacement for oversampling
            )
            
            # Update the train dataloader with the new sampler
            train_dataloader = runner.train_dataloader
            
            # Create new dataloader with weighted sampler
            from torch.utils.data import DataLoader
            
            new_train_dataloader = DataLoader(
                dataset=train_dataloader.dataset,
                batch_size=train_dataloader.batch_size,
                sampler=weighted_sampler,
                num_workers=train_dataloader.num_workers,
                pin_memory=getattr(train_dataloader, 'pin_memory', False),
                drop_last=getattr(train_dataloader, 'drop_last', False),
                persistent_workers=getattr(train_dataloader, 'persistent_workers', False),
            )
            
            # Replace the runner's train dataloader
            runner.train_dataloader = new_train_dataloader
            
            num_hard_examples = np.sum(sample_weights > 1.0)
            logger.info(f'[ConcurrentMLPTrainingHook] Updated HRNet training with weighted sampler')
            logger.info(f'[ConcurrentMLPTrainingHook] Hard examples will be oversampled {self.hrnet_hard_example_weight}x for next epoch')
            logger.info(f'[ConcurrentMLPTrainingHook] {num_hard_examples}/{len(sample_weights)} samples marked as hard examples')
            
        except Exception as e:
            logger.warning(f'[ConcurrentMLPTrainingHook] Failed to create weighted sampler for HRNet: {e}')
            logger.warning('[ConcurrentMLPTrainingHook] Continuing with standard sampling for HRNet')

    def before_train_epoch(self, runner: Runner):
        """Apply weighted sampling for HRNetV2 training if hard examples were identified."""
        logger: MMLogger = runner.logger
        
        # Skip first epoch (no hard examples identified yet)
        if runner.epoch == 0:
            logger.info('[ConcurrentMLPTrainingHook] First epoch - using standard sampling for HRNet')
            return
            
        # Apply weighted sampling if we have sample weights from previous epoch
        if self.sample_weights_for_hrnet is not None:
            logger.info(f'[ConcurrentMLPTrainingHook] Applying hard-example oversampling for HRNet epoch {runner.epoch + 1}')
            self._create_weighted_sampler_for_hrnet(runner, self.sample_weights_for_hrnet)
        else:
            logger.info('[ConcurrentMLPTrainingHook] No hard examples identified yet - using standard sampling')

    def _compute_class_from_anb(self, landmarks: np.ndarray, landmark_names: List[str]) -> int:
        """Compute class from ANB angle calculated from landmarks.
        
        Args:
            landmarks: [19, 2] array of landmark coordinates
            landmark_names: List of landmark names
            
        Returns:
            int: Class label (0=Class I, 1=Class II, 2=Class III)
        """
        # Create landmark index map
        landmark_idx = {name: i for i, name in enumerate(landmark_names)}
        
        # Get required landmarks
        try:
            sella = landmarks[landmark_idx['sella']]
            nasion = landmarks[landmark_idx['nasion']]
            a_point = landmarks[landmark_idx['A_point']]
            b_point = landmarks[landmark_idx['B_point']]
        except KeyError:
            # If landmarks not found, default to Class I
            return 0
        
        # Calculate angles
        def angle(p1, p2, p3):
            """Calculate angle at p2 between vectors p2->p1 and p2->p3."""
            v1 = p1 - p2
            v2 = p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            return np.degrees(np.arccos(cos_angle))
        
        # Calculate SNA and SNB angles
        sna = angle(sella, nasion, a_point)
        snb = angle(sella, nasion, b_point)
        
        # Calculate ANB angle
        anb = sna - snb
        
        # Classify based on ANB angle
        if anb > 4:
            return 1  # Class II
        elif anb < 2:
            return 2  # Class III
        else:
            return 0  # Class I

    def after_train_epoch(self, runner: Runner):
        """After each HRNetV2 epoch, train joint MLP on-the-fly using current predictions."""
        logger: MMLogger = runner.logger
        assert self.mlp_joint is not None

        # -----------------------------------------------------------------
        # Step 1: Generate predictions on training data (GPU-optimized with batching)
        # -----------------------------------------------------------------
        
        # Get the actual model, handling potential wrapping
        model = runner.model
        if hasattr(model, 'module'):
            # Handle DDP or other wrapped models
            actual_model = model.module
        else:
            actual_model = model
            
        # Ensure model has required attributes for inference
        if not hasattr(actual_model, 'cfg') and hasattr(runner, 'cfg'):
            actual_model.cfg = runner.cfg
            
        # Set up dataset_meta if missing (required for inference_topdown)
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
        all_preds: List[np.ndarray] = []
        all_gts: List[np.ndarray] = []
        all_errors: List[np.ndarray] = []  # For hard-example detection
        all_gt_classes: List[int] = []  # Ground truth classes
        
        # Batch processing parameters
        BATCH_SIZE = 80  # Process 80 images at once for speed
        
        train_dataset = runner.train_dataloader.dataset
        logger.info(f'[ConcurrentMLPTrainingHook] Generating predictions for joint MLP with batch size {BATCH_SIZE}...')

        def tensor_to_numpy(data):
            """Safely convert tensor to numpy, handling both tensor and numpy inputs."""
            if isinstance(data, torch.Tensor):
                return data.cpu().numpy()
            elif isinstance(data, np.ndarray):
                return data
            else:
                return np.array(data)

        def batch_inference(images_batch, gt_keypoints_batch, gt_classes_batch):
            """Run batched inference on a list of images."""
            batch_preds = []
            batch_gts = []
            batch_errors = []
            batch_gt_classes = []
            
            try:
                # Process each image individually but in a batch-like manner
                for i, (img, gt_kpts, gt_class) in enumerate(zip(images_batch, gt_keypoints_batch, gt_classes_batch)):
                    if img is not None and gt_kpts is not None:
                        try:
                            # Run inference on individual image
                            bbox = np.array([[0, 0, 224, 224]], dtype=np.float32)
                            
                            with torch.no_grad():
                                results = inference_topdown(model, img, bboxes=bbox, bbox_format='xyxy')
                            
                            if results and len(results) > 0:
                                pred_kpts = results[0].pred_instances.keypoints[0]
                                pred_kpts = tensor_to_numpy(pred_kpts)
                                
                                if pred_kpts is not None and pred_kpts.shape[0] == 19:
                                    # Flatten coordinates to 38-D vectors
                                    pred_flat = pred_kpts.flatten()
                                    gt_flat = gt_kpts.flatten()
                                    
                                    # Calculate per-landmark radial errors
                                    landmark_errors = np.sqrt(np.sum((pred_kpts - gt_kpts)**2, axis=1))
                                    
                                    batch_preds.append(pred_flat)
                                    batch_gts.append(gt_flat)
                                    batch_errors.append(landmark_errors)
                                    batch_gt_classes.append(gt_class)
                                    
                        except Exception as e:
                            # Skip failed individual images
                            continue
                             
            except Exception as e:
                logger.warning(f'[ConcurrentMLPTrainingHook] Batch processing failed: {e}')
                
            return batch_preds, batch_gts, batch_errors, batch_gt_classes

        # Access training data more robustly with batching
        try:
            from tqdm import tqdm
            
            # Get landmark names for ANB calculation
            import cephalometric_dataset_info
            landmark_names = cephalometric_dataset_info.landmark_names_in_order
            
            # Method 1: Try to access the raw annotation file
            if hasattr(train_dataset, 'ann_file') and train_dataset.ann_file:
                logger.info(f'[ConcurrentMLPTrainingHook] Loading data from annotation file: {train_dataset.ann_file}')
                
                try:
                    df = pd.read_json(train_dataset.ann_file)
                    logger.info(f'[ConcurrentMLPTrainingHook] Loaded {len(df)} samples from annotation file')
                    
                    # Import landmark info
                    landmark_cols = cephalometric_dataset_info.original_landmark_cols
                    
                    processed_count = 0
                    
                    # Process in batches
                    for batch_start in tqdm(range(0, len(df), BATCH_SIZE), disable=runner.rank != 0, desc="Batch Inference from File"):
                        batch_end = min(batch_start + BATCH_SIZE, len(df))
                        batch_df = df.iloc[batch_start:batch_end]
                        
                        # Prepare batch data
                        images_batch = []
                        gt_keypoints_batch = []
                        gt_classes_batch = []
                        
                        for idx, row in batch_df.iterrows():
                            try:
                                # Get image from row
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
                                
                                if valid_gt:
                                    gt_keypoints = np.array(gt_keypoints)
                                    
                                    # Get ground truth class
                                    if 'class' in row and pd.notna(row['class']):
                                        gt_class = int(row['class']) - 1  # Convert to 0-indexed
                                    else:
                                        # Compute from ANB angle
                                        gt_class = self._compute_class_from_anb(gt_keypoints, landmark_names)
                                    
                                    images_batch.append(img_array)
                                    gt_keypoints_batch.append(gt_keypoints)
                                    gt_classes_batch.append(gt_class)
                                else:
                                    images_batch.append(None)
                                    gt_keypoints_batch.append(None)
                                    gt_classes_batch.append(None)
                                    
                            except Exception as e:
                                logger.warning(f'[ConcurrentMLPTrainingHook] Failed to prepare sample {idx}: {e}')
                                images_batch.append(None)
                                gt_keypoints_batch.append(None)
                                gt_classes_batch.append(None)
                                continue
                        
                        # Run batch inference
                        batch_preds, batch_gts, batch_errors, batch_gt_classes = batch_inference(
                            images_batch, gt_keypoints_batch, gt_classes_batch)
                        
                        # Add to results
                        all_preds.extend(batch_preds)
                        all_gts.extend(batch_gts)
                        all_errors.extend(batch_errors)
                        all_gt_classes.extend(batch_gt_classes)
                        
                        processed_count += len(batch_preds)
                    
                    logger.info(f'[ConcurrentMLPTrainingHook] Successfully processed {processed_count} samples from file using batch inference')
                    
                except Exception as e:
                    logger.warning(f'[ConcurrentMLPTrainingHook] Failed to load from annotation file: {e}')
                    df = None
            
            # Method 2: Fallback to dataset iteration if file method fails
            if not all_preds:  # No predictions from file method
                logger.info('[ConcurrentMLPTrainingHook] Using dataset iteration method with batching')
                
                processed_count = 0
                
                # Process in batches
                for batch_start in tqdm(range(0, len(train_dataset), BATCH_SIZE), disable=runner.rank != 0, desc="Batch Dataset Inference"):
                    batch_end = min(batch_start + BATCH_SIZE, len(train_dataset))
                    
                    # Prepare batch data
                    images_batch = []
                    gt_keypoints_batch = []
                    gt_classes_batch = []
                    
                    for idx in range(batch_start, batch_end):
                        try:
                            data_sample = train_dataset[idx]
                            
                            # Extract image - handle different possible formats
                            if 'inputs' in data_sample:
                                img = data_sample['inputs']
                                if isinstance(img, torch.Tensor):
                                    # Convert from (C, H, W) tensor to (H, W, C) numpy
                                    img_np = tensor_to_numpy(img.permute(1, 2, 0) * 255).astype(np.uint8)
                                else:
                                    img_np = np.array(img, dtype=np.uint8)
                            else:
                                images_batch.append(None)
                                gt_keypoints_batch.append(None)
                                gt_classes_batch.append(None)
                                continue
                            
                            # Extract ground truth keypoints with robust handling
                            if 'data_samples' in data_sample and hasattr(data_sample['data_samples'], 'gt_instances'):
                                gt_instances = data_sample['data_samples'].gt_instances
                                if hasattr(gt_instances, 'keypoints') and len(gt_instances.keypoints) > 0:
                                    gt_kpts = tensor_to_numpy(gt_instances.keypoints[0])
                                    
                                    # Extract ground truth class
                                    if hasattr(gt_instances, 'labels'):
                                        gt_class = int(gt_instances.labels[0])
                                    elif hasattr(data_sample['data_samples'], 'metainfo') and 'class' in data_sample['data_samples'].metainfo:
                                        gt_class = int(data_sample['data_samples'].metainfo['class']) - 1  # Convert to 0-indexed
                                    else:
                                        # Compute from ANB angle
                                        gt_class = self._compute_class_from_anb(gt_kpts, landmark_names)
                                    
                                    images_batch.append(img_np)
                                    gt_keypoints_batch.append(gt_kpts)
                                    gt_classes_batch.append(gt_class)
                                else:
                                    images_batch.append(None)
                                    gt_keypoints_batch.append(None)
                                    gt_classes_batch.append(None)
                            else:
                                images_batch.append(None)
                                gt_keypoints_batch.append(None)
                                gt_classes_batch.append(None)
                                
                        except Exception as e:
                            # Only log every 100th error to avoid spam
                            if idx % 100 == 0:
                                logger.warning(f'[ConcurrentMLPTrainingHook] Failed to process sample {idx}: {e}')
                            images_batch.append(None)
                            gt_keypoints_batch.append(None)
                            gt_classes_batch.append(None)
                            continue
                    
                    # Run batch inference
                    batch_preds, batch_gts, batch_errors, batch_gt_classes = batch_inference(
                        images_batch, gt_keypoints_batch, gt_classes_batch)
                    
                    # Add to results
                    all_preds.extend(batch_preds)
                    all_gts.extend(batch_gts)
                    all_errors.extend(batch_errors)
                    all_gt_classes.extend(batch_gt_classes)
                    
                    processed_count += len(batch_preds)
                
                logger.info(f'[ConcurrentMLPTrainingHook] Successfully processed {processed_count} samples via dataset iteration with batch inference')

        except Exception as e:
            logger.error(f'[ConcurrentMLPTrainingHook] Critical error during batch data processing: {e}')
            return

        if not all_preds:
            logger.warning('[ConcurrentMLPTrainingHook] No predictions generated; skipping MLP update.')
            return

        all_preds = np.stack(all_preds)  # [N, 38]
        all_gts = np.stack(all_gts)      # [N, 38]
        all_errors = np.stack(all_errors)  # [N, 19]
        all_gt_classes = np.array(all_gt_classes)  # [N]
        
        logger.info(f'[ConcurrentMLPTrainingHook] Generated predictions for {len(all_preds)} samples using batch inference (batch_size={BATCH_SIZE})')
        
        # Log class distribution
        unique_classes, class_counts = np.unique(all_gt_classes, return_counts=True)
        logger.info(f'[ConcurrentMLPTrainingHook] Class distribution: {dict(zip(unique_classes, class_counts))}')

        # -----------------------------------------------------------------
        # Step 1.5: Compute hard-example weights
        # -----------------------------------------------------------------
        # Determine sample weights based on maximum landmark error per sample
        max_errors_per_sample = np.max(all_errors, axis=1)  # [N,] - worst landmark per sample
        hard_examples = max_errors_per_sample > self.hard_example_threshold
        
        # Create sample weights: 1.0 for normal, hrnet_hard_example_weight for hard examples
        sample_weights = np.ones(len(all_preds))
        sample_weights[hard_examples] = self.hrnet_hard_example_weight
        
        num_hard_examples = np.sum(hard_examples)
        logger.info(f'[ConcurrentMLPTrainingHook] Hard examples (>{self.hard_example_threshold}px): {num_hard_examples}/{len(all_preds)} ({num_hard_examples/len(all_preds)*100:.1f}%)')
        
        if num_hard_examples > 0:
            logger.info(f'[ConcurrentMLPTrainingHook] Hard example errors: min={np.min(max_errors_per_sample[hard_examples]):.2f}, max={np.max(max_errors_per_sample[hard_examples]):.2f}, mean={np.mean(max_errors_per_sample[hard_examples]):.2f}')
            logger.info(f'[ConcurrentMLPTrainingHook] Hard examples will be weighted {self.hrnet_hard_example_weight}x for both MLP and next HRNet epoch')

        # Store sample weights for next HRNet epoch
        self.sample_weights_for_hrnet = sample_weights.copy()
        logger.info(f'[ConcurrentMLPTrainingHook] Sample weights stored for next HRNet epoch training')

        # -----------------------------------------------------------------
        # Step 2: Initialize normalization scalers (only once)
        # -----------------------------------------------------------------
        if not self.scalers_initialized:
            logger.info('[ConcurrentMLPTrainingHook] Initializing normalization scalers for 38-D data...')
            
            # Fit scalers on the first batch of data
            self.scaler_input.fit(all_preds)
            self.scaler_target.fit(all_gts)
            
            self.scalers_initialized = True
            logger.info('[ConcurrentMLPTrainingHook] Normalization scalers initialized')
            
            # Save scalers for evaluation
            save_dir = os.path.join(runner.work_dir, 'concurrent_mlp')
            os.makedirs(save_dir, exist_ok=True)
            
            joblib.dump(self.scaler_input, os.path.join(save_dir, 'scaler_joint_input.pkl'))
            joblib.dump(self.scaler_target, os.path.join(save_dir, 'scaler_joint_target.pkl'))
            
            logger.info(f'[ConcurrentMLPTrainingHook] Joint scalers saved to {save_dir}')

        # Normalize data using the consistent scalers
        preds_scaled = self.scaler_input.transform(all_preds)
        gts_scaled = self.scaler_target.transform(all_gts)

        # -----------------------------------------------------------------
        # Step 3: Train joint MLP for fixed number of epochs (GPU-optimized)
        # -----------------------------------------------------------------
        logger.info('[ConcurrentMLPTrainingHook] Training joint 38-D MLP with classification on GPU…')
        
        # Calculate initial loss before refinement for logging
        initial_regression_loss = self.criterion_regression(
            torch.from_numpy(preds_scaled).float(), 
            torch.from_numpy(gts_scaled).float()
        ).item()

        # Build dataset with hard-example oversampling
        ds_joint = _MLPDataset(preds_scaled, gts_scaled, all_gt_classes, sample_weights)
        dl_joint = data.DataLoader(ds_joint, batch_size=self.mlp_batch_size, shuffle=True, pin_memory=True)

        def _train_joint(model: JointMLPRefinementModel, optimiser: optim.Optimizer, loader: data.DataLoader, initial_loss: float):
            model.train()
            total_loss = 0.0
            for ep in range(self.mlp_epochs):
                epoch_loss = 0.0
                epoch_regression_loss = 0.0
                epoch_classification_loss = 0.0
                correct_predictions = 0
                total_predictions = 0
                selection_weights_sum = torch.zeros(38).to(self.device)
                selection_weights_count = 0
                
                for preds_batch, gts_batch, gt_classes_batch in loader:
                    preds_batch = preds_batch.to(self.device, non_blocking=True)
                    gts_batch = gts_batch.to(self.device, non_blocking=True)
                    gt_classes_batch = gt_classes_batch.to(self.device, non_blocking=True)

                    optimiser.zero_grad()
                    
                    # Forward pass
                    outputs, class_logits = model(preds_batch)
                    
                    # Calculate losses
                    regression_loss = self.criterion_regression(outputs, gts_batch)
                    classification_loss = self.criterion_classification(class_logits, gt_classes_batch)
                    
                    # Combined loss
                    loss = regression_loss + self.classification_loss_weight * classification_loss
                    
                    loss.backward()
                    optimiser.step()

                    epoch_loss += loss.item()
                    epoch_regression_loss += regression_loss.item()
                    epoch_classification_loss += classification_loss.item()
                    
                    # Track classification accuracy
                    pred_classes = class_logits.argmax(dim=1)
                    correct_predictions += (pred_classes == gt_classes_batch).sum().item()
                    total_predictions += gt_classes_batch.size(0)
                    
                    # Track selection weights
                    if hasattr(model, 'last_selection_weights'):
                        selection_weights_sum += model.last_selection_weights.detach().sum(dim=0)
                        selection_weights_count += model.last_selection_weights.shape[0]
                
                total_loss = epoch_loss / len(loader)
                avg_regression_loss = epoch_regression_loss / len(loader)
                avg_classification_loss = epoch_classification_loss / len(loader)
                classification_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                
                # Calculate average selection weights
                if selection_weights_count > 0:
                    avg_selection_weights = selection_weights_sum / selection_weights_count
                    avg_weight_use_mlp = avg_selection_weights.mean().item()
                    
                if (ep + 1) % 20 == 0:
                    improvement = ((initial_loss - avg_regression_loss) / initial_loss * 100) if initial_loss > 0 else 0
                    if selection_weights_count > 0:
                        logger.info(f'[ConcurrentMLPTrainingHook] Joint-MLP epoch {ep+1}/{self.mlp_epochs} | '
                                  f'Total loss: {total_loss:.6f} | Regression: {avg_regression_loss:.6f} ({improvement:+.1f}%) | '
                                  f'Classification: {avg_classification_loss:.6f} (Acc: {classification_accuracy:.3f}) | '
                                  f'Avg MLP usage: {avg_weight_use_mlp:.3f}')
                        
                        # Log per-coordinate selection preference every 40 epochs
                        if (ep + 1) % 40 == 0:
                            # Convert to per-landmark statistics (19 landmarks × 2 coords)
                            landmark_weights = avg_selection_weights.view(19, 2).mean(dim=1)
                            max_mlp_landmarks = torch.topk(landmark_weights, k=5).indices
                            min_mlp_landmarks = torch.topk(landmark_weights, k=5, largest=False).indices
                            logger.info(f'[ConcurrentMLPTrainingHook] Top 5 landmarks preferring MLP: {max_mlp_landmarks.tolist()} (weights: {landmark_weights[max_mlp_landmarks].tolist()})')
                            logger.info(f'[ConcurrentMLPTrainingHook] Top 5 landmarks preferring HRNet: {min_mlp_landmarks.tolist()} (weights: {landmark_weights[min_mlp_landmarks].tolist()})')
                    else:
                        logger.info(f'[ConcurrentMLPTrainingHook] Joint-MLP epoch {ep+1}/{self.mlp_epochs} | '
                                  f'Total loss: {total_loss:.6f} | Regression: {avg_regression_loss:.6f} ({improvement:+.1f}%) | '
                                  f'Classification: {avg_classification_loss:.6f} (Acc: {classification_accuracy:.3f})')

        _train_joint(self.mlp_joint, self.opt_joint, dl_joint, initial_regression_loss)

        logger.info('[ConcurrentMLPTrainingHook] Finished joint MLP update for this HRNet epoch.')
        
        # Save MLP models after each epoch
        save_dir = os.path.join(runner.work_dir, 'concurrent_mlp')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save current epoch models
        current_epoch = runner.epoch + 1  # runner.epoch is 0-indexed
        mlp_joint_epoch_path = os.path.join(save_dir, f'mlp_joint_epoch_{current_epoch}.pth')
        
        torch.save(self.mlp_joint.state_dict(), mlp_joint_epoch_path)
        
        # Also save as "latest" for easy access
        mlp_joint_latest_path = os.path.join(save_dir, 'mlp_joint_latest.pth')
        torch.save(self.mlp_joint.state_dict(), mlp_joint_latest_path)
        
        logger.info(f'[ConcurrentMLPTrainingHook] Joint MLP model saved for epoch {current_epoch}')
        logger.info(f'[ConcurrentMLPTrainingHook] Latest model: {mlp_joint_latest_path}')

    # ---------------------------------------------------------------------
    # Optional: save MLP weights at end of run
    # ---------------------------------------------------------------------
    def after_run(self, runner: Runner):
        logger: MMLogger = runner.logger
        if self.mlp_joint is None:
            return
        save_dir = os.path.join(runner.work_dir, 'concurrent_mlp')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.mlp_joint.state_dict(), os.path.join(save_dir, 'mlp_joint_final.pth'))
        logger.info(f'[ConcurrentMLPTrainingHook] Saved final joint MLP weights to {save_dir}')

    def _save_synchronized_mlp_model(self, runner: Runner, checkpoint_name: str):
        """Save MLP model synchronized with a specific HRNet checkpoint."""
        logger: MMLogger = runner.logger
        
        if self.mlp_joint is None:
            return
            
        try:
            save_dir = os.path.join(runner.work_dir, 'concurrent_mlp')
            os.makedirs(save_dir, exist_ok=True)
            
            # Create synchronized MLP checkpoint name
            # Extract meaningful part from HRNet checkpoint name
            if checkpoint_name.endswith('.pth'):
                base_name = checkpoint_name[:-4]  # Remove .pth extension
            else:
                base_name = checkpoint_name
                
            synchronized_mlp_path = os.path.join(save_dir, f'mlp_joint_sync_{base_name}.pth')
            
            # Save MLP model
            torch.save(self.mlp_joint.state_dict(), synchronized_mlp_path)
            
            # Update mapping
            self.checkpoint_mlp_mapping[checkpoint_name] = synchronized_mlp_path
            
            # Save mapping to file for evaluation scripts
            mapping_file = os.path.join(save_dir, 'checkpoint_mlp_mapping.json')
            import json
            with open(mapping_file, 'w') as f:
                json.dump(self.checkpoint_mlp_mapping, f, indent=2)
            
            logger.info(f'[ConcurrentMLPTrainingHook] Synchronized MLP model saved: {os.path.basename(synchronized_mlp_path)}')
            logger.info(f'[ConcurrentMLPTrainingHook] Mapped to HRNet checkpoint: {checkpoint_name}')
            
        except Exception as e:
            logger.warning(f'[ConcurrentMLPTrainingHook] Failed to save synchronized MLP model: {e}')

    def after_save_checkpoint(self, runner: Runner, checkpoint_path: str):
        """Hook called after HRNet checkpoint is saved - save synchronized MLP model."""
        logger: MMLogger = runner.logger
        
        # Extract checkpoint filename
        checkpoint_name = os.path.basename(checkpoint_path)
        
        # Only save synchronized models for important checkpoints
        important_checkpoints = ['best_', 'latest', 'epoch_']
        is_important = any(checkpoint_name.startswith(prefix) for prefix in important_checkpoints)
        
        if is_important and self.mlp_joint is not None:
            logger.info(f'[ConcurrentMLPTrainingHook] HRNet checkpoint saved: {checkpoint_name}')
            self._save_synchronized_mlp_model(runner, checkpoint_name)
            self.last_saved_checkpoint = checkpoint_name 