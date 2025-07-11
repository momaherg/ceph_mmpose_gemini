#!/usr/bin/env python3
"""
Multi-task Cephalometric Model: HRNetV2 with Landmark Detection + Classification
This model performs simultaneous landmark detection and patient classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
from mmpose.models.builder import MODELS
from mmpose.models.pose_estimators import TopdownPoseEstimator
from mmengine.structures import InstanceData, PixelData
from mmpose.structures import PoseDataSample


@MODELS.register_module()
class MultiTaskCephalometricModel(TopdownPoseEstimator):
    """Multi-task model for cephalometric landmark detection and patient classification.
    
    This model extends HRNetV2 to perform:
    1. Landmark detection via heatmaps (19 landmarks)
    2. Patient classification (Class I, II, III)
    
    The classification can be predicted:
    - Directly from backbone features (native classification)
    - From predicted landmarks (ANB-based, for comparison)
    """
    
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 classification_head=None,
                 classification_loss_weight=1.0,
                 use_landmark_features_for_classification=True,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        """Initialize multi-task model.
        
        Args:
            backbone: Config for backbone network
            neck: Config for neck network
            head: Config for landmark heatmap head
            classification_head: Config for classification head
            classification_loss_weight: Weight for classification loss
            use_landmark_features_for_classification: If True, use both backbone features 
                and predicted landmark coordinates for classification
            train_cfg: Training config
            test_cfg: Test config
            data_preprocessor: Data preprocessor config
            init_cfg: Initialization config
        """
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg
        )
        
        self.classification_loss_weight = classification_loss_weight
        self.use_landmark_features_for_classification = use_landmark_features_for_classification
        
        # Build classification head
        if classification_head is None:
            # Default classification head configuration
            classification_head = dict(
                type='ClassificationHead',
                in_channels=270,  # HRNet-W18 concatenated features: 18+36+72+144
                num_classes=3,    # Class I, II, III
                hidden_dim=256,
                dropout_rate=0.2
            )
        
        # If using landmark features, we'll use the fusion output dimension
        if self.use_landmark_features_for_classification:
            classification_head_in_channels = 256  # Output of fusion module
        else:
            classification_head_in_channels = classification_head.get('in_channels', 270)
        
        # Update classification head with correct input channels
        classification_head['in_channels'] = classification_head_in_channels
        
        self.classification_head = MODELS.build(classification_head)
        
        # If using landmark features, create a fusion module
        if self.use_landmark_features_for_classification:
            # Fusion module to combine backbone features with landmark coordinates
            self.landmark_feature_extractor = nn.Sequential(
                nn.Linear(38, 128),  # 19 landmarks * 2 coords = 38
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(128, 64)
            )
            
            # Update classification head input dimension
            self.classification_fusion = nn.Sequential(
                nn.Linear(classification_head.get('in_channels', 270) + 64, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            )
    
    def extract_feat(self, inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        """Extract features from backbone."""
        x = self.backbone(inputs)
        return x
    
    def loss(self, inputs: torch.Tensor, data_samples: List[PoseDataSample]) -> dict:
        """Calculate losses from a batch of inputs and data samples.
        
        Args:
            inputs: Input images
            data_samples: Data samples containing ground truth
            
        Returns:
            dict: Losses including landmark loss and classification loss
        """
        # Extract features
        feats = self.extract_feat(inputs)
        
        # Process features through neck if available
        if self.neck is not None:
            neck_feats = self.neck(feats)
        else:
            neck_feats = feats
        
        # Landmark detection loss (from heatmap head)
        losses = self.head.loss(neck_feats, data_samples)
        
        # Get features for classification
        if self.neck is not None:
            # Use neck output features
            classification_features = neck_feats
            if isinstance(classification_features, (list, tuple)):
                # Concatenate multi-scale features
                classification_features = torch.cat([F.adaptive_avg_pool2d(f, 1).flatten(1) 
                                                    for f in classification_features], dim=1)
            else:
                # Neck output is a single tensor with spatial dimensions
                classification_features = F.adaptive_avg_pool2d(classification_features, 1).flatten(1)
        else:
            # Use backbone output directly
            if isinstance(feats, (list, tuple)):
                # Concatenate multi-scale features
                classification_features = torch.cat([F.adaptive_avg_pool2d(f, 1).flatten(1) 
                                                    for f in feats], dim=1)
            else:
                classification_features = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        
        # If using landmark features for classification
        if self.use_landmark_features_for_classification:
            # Get predicted landmarks from heatmaps
            with torch.no_grad():
                pred_heatmaps = self.head.forward(neck_feats)
                # Decode heatmaps to coordinates
                pred_coords = self._decode_heatmaps_to_coords(pred_heatmaps)
                # Flatten coordinates: [batch, 19, 2] -> [batch, 38]
                pred_coords_flat = pred_coords.flatten(1)
            
            # Extract landmark features
            landmark_features = self.landmark_feature_extractor(pred_coords_flat)
            
            # Fuse backbone features with landmark features
            fused_features = torch.cat([classification_features, landmark_features], dim=1)
            classification_features = self.classification_fusion(fused_features)
        
        # Get ground truth classes
        gt_classes = []
        for data_sample in data_samples:
            # Extract class from data sample
            if hasattr(data_sample, 'gt_instances') and hasattr(data_sample.gt_instances, 'labels'):
                gt_class = data_sample.gt_instances.labels
            else:
                # Try to get from metainfo
                gt_class = data_sample.metainfo.get('class', 1)  # Default to Class I
                # Convert to 0-indexed (Class I=0, II=1, III=2)
                gt_class = int(gt_class) - 1
                gt_class = torch.tensor(gt_class, device=inputs.device)
            gt_classes.append(gt_class)
        
        gt_classes = torch.stack(gt_classes)
        
        # Classification prediction and loss
        class_logits = self.classification_head(classification_features)
        classification_loss = F.cross_entropy(class_logits, gt_classes)
        
        # Add classification loss to total losses
        losses['loss_classification'] = classification_loss * self.classification_loss_weight
        
        # Add classification accuracy for monitoring
        with torch.no_grad():
            pred_classes = class_logits.argmax(dim=1)
            accuracy = (pred_classes == gt_classes).float().mean()
            losses['acc_classification'] = accuracy
        
        return losses
    
    def predict(self, inputs: torch.Tensor, data_samples: List[PoseDataSample]) -> List[PoseDataSample]:
        """Predict results from a batch of inputs and data samples.
        
        Args:
            inputs: Input images
            data_samples: Data samples
            
        Returns:
            List[PoseDataSample]: Predictions with both landmarks and classification
        """
        # Get landmark predictions from parent class
        data_samples = super().predict(inputs, data_samples)
        
        # Extract features
        feats = self.extract_feat(inputs)
        
        # Process features through neck if available
        if self.neck is not None:
            neck_feats = self.neck(feats)
        else:
            neck_feats = feats
        
        # Get features for classification
        if self.neck is not None:
            # Use neck output features
            classification_features = neck_feats
            if isinstance(classification_features, (list, tuple)):
                # Concatenate multi-scale features
                classification_features = torch.cat([F.adaptive_avg_pool2d(f, 1).flatten(1) 
                                                    for f in classification_features], dim=1)
            else:
                # Neck output is a single tensor with spatial dimensions
                classification_features = F.adaptive_avg_pool2d(classification_features, 1).flatten(1)
        else:
            # Use backbone output directly
            if isinstance(feats, (list, tuple)):
                # Concatenate multi-scale features
                classification_features = torch.cat([F.adaptive_avg_pool2d(f, 1).flatten(1) 
                                                    for f in feats], dim=1)
            else:
                classification_features = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        
        # If using landmark features for classification
        if self.use_landmark_features_for_classification:
            # Get predicted landmarks
            pred_coords_list = []
            for data_sample in data_samples:
                if hasattr(data_sample, 'pred_instances') and hasattr(data_sample.pred_instances, 'keypoints'):
                    pred_coords = torch.from_numpy(data_sample.pred_instances.keypoints).to(inputs.device)
                    pred_coords_list.append(pred_coords.flatten())
            
            if pred_coords_list:
                pred_coords_flat = torch.stack(pred_coords_list)
                landmark_features = self.landmark_feature_extractor(pred_coords_flat)
                fused_features = torch.cat([classification_features, landmark_features], dim=1)
                classification_features = self.classification_fusion(fused_features)
        
        # Predict classes
        class_logits = self.classification_head(classification_features)
        class_probs = F.softmax(class_logits, dim=1)
        pred_classes = class_logits.argmax(dim=1)
        
        # Add classification results to data samples
        for i, data_sample in enumerate(data_samples):
            # Add classification prediction
            if not hasattr(data_sample, 'pred_instances'):
                data_sample.pred_instances = InstanceData()
            
            # Store both class (0-indexed) and probabilities
            data_sample.pred_instances.class_label = pred_classes[i].cpu().numpy()
            data_sample.pred_instances.class_probs = class_probs[i].cpu().numpy()
            
            # Also store human-readable class name
            class_names = ['Class I', 'Class II', 'Class III']
            data_sample.pred_instances.class_name = class_names[pred_classes[i].item()]
        
        return data_samples
    
    def _decode_heatmaps_to_coords(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """Decode heatmaps to coordinates using simple argmax.
        
        Args:
            heatmaps: Heatmaps tensor [batch, num_joints, height, width]
            
        Returns:
            torch.Tensor: Coordinates [batch, num_joints, 2]
        """
        batch_size, num_joints, h, w = heatmaps.shape
        
        # Reshape heatmaps for processing
        heatmaps_reshaped = heatmaps.view(batch_size, num_joints, -1)
        
        # Get max indices
        max_indices = heatmaps_reshaped.argmax(dim=2)
        
        # Convert to x, y coordinates
        y_coords = (max_indices // w).float()
        x_coords = (max_indices % w).float()
        
        # Normalize to [0, 1] range
        x_coords = x_coords / (w - 1)
        y_coords = y_coords / (h - 1)
        
        # Stack coordinates
        coords = torch.stack([x_coords, y_coords], dim=2)
        
        # Scale to input image size (assuming 224x224)
        coords = coords * 224
        
        return coords


@MODELS.register_module()
class ClassificationHead(nn.Module):
    """Classification head for patient classification.
    
    Simple MLP head that takes feature vectors and outputs class logits.
    """
    
    def __init__(self,
                 in_channels: int,
                 num_classes: int = 3,
                 hidden_dim: int = 256,
                 dropout_rate: float = 0.2):
        """Initialize classification head.
        
        Args:
            in_channels: Input feature dimension
            num_classes: Number of classes (default: 3 for Class I, II, III)
            hidden_dim: Hidden layer dimension
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input features [batch, in_channels]
            
        Returns:
            torch.Tensor: Class logits [batch, num_classes]
        """
        return self.classifier(x)
    
    def get(self, key: str, default=None):
        """Get attribute for config compatibility."""
        return getattr(self, key, default) 