"""
HRNetV2 model with additional classification head for skeletal pattern prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpose.registry import MODELS  # Changed from mmpose.models import MODELS
from mmpose.models.heads import HeatmapHead
from mmengine.model import BaseModule
from typing import Optional, Sequence, Tuple, Union


@MODELS.register_module()
class HRNetV2WithClassification(HeatmapHead):
    """HRNetV2 head with additional classification output for skeletal patterns.
    
    This model extends the standard HeatmapHead to also predict skeletal
    classification (Class I, II, or III) based on the feature representations.
    
    Args:
        num_classes (int): Number of skeletal pattern classes (default: 3)
        classification_hidden_dim (int): Hidden dimension for classification head (default: 256)
        classification_dropout (float): Dropout rate for classification head (default: 0.2)
        classification_loss (dict): Config for classification loss (default: CrossEntropyLoss)
        classification_loss_weight (float): Weight for classification loss (default: 1.0)
        **kwargs: Additional arguments for HeatmapHead
    """
    
    def __init__(self,
                 num_classes: int = 3,
                 classification_hidden_dim: int = 256,
                 classification_dropout: float = 0.2,
                 classification_loss: dict = dict(type='CrossEntropyLoss'),
                 classification_loss_weight: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.num_classes = num_classes
        self.classification_loss_weight = classification_loss_weight
        
        # Build classification head
        # We'll use the same input channels as the heatmap head
        in_channels = kwargs.get('in_channels', 32)
        
        self.classification_head = nn.Sequential(
            # Global average pooling to aggregate spatial features
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # Classification layers
            nn.Linear(in_channels, classification_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(classification_dropout),
            nn.Linear(classification_hidden_dim, classification_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(classification_dropout),
            nn.Linear(classification_hidden_dim // 2, num_classes)
        )
        
        # Build classification loss - use PyTorch's CrossEntropyLoss directly
        self.classification_loss = nn.CrossEntropyLoss()
        
    def forward(self, feats: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function.
        
        Args:
            feats (Tuple[Tensor]): Multi-level features from backbone
            
        Returns:
            Tuple[Tensor, Tensor]: (heatmaps, classification_logits)
                - heatmaps: shape (N, K, H, W) for K keypoints
                - classification_logits: shape (N, num_classes)
        """
        # Get heatmaps from parent class
        heatmaps = super().forward(feats)
        
        # Get classification from the highest resolution feature map
        # feats is a tuple of feature maps at different scales
        # We use the last (highest resolution) feature map
        x = feats[-1] if isinstance(feats, (list, tuple)) else feats
        
        # Pass through classification head
        classification_logits = self.classification_head(x)
        
        return heatmaps, classification_logits
    
    def predict(self,
                feats: Tuple[torch.Tensor],
                batch_data_samples,
                test_cfg: Optional[dict] = None):
        """Predict function for inference.
        
        Args:
            feats (Tuple[Tensor]): Multi-level features from backbone
            batch_data_samples: Batch of data samples
            test_cfg (dict, optional): Test configuration
            
        Returns:
            InstanceList: Predictions with both keypoints and classification
        """
        # Get heatmaps and classification logits
        heatmaps, classification_logits = self.forward(feats)
        
        # Decode heatmaps to keypoints using parent class method
        preds = super().predict(feats, batch_data_samples, test_cfg)
        
        # Add classification predictions
        classification_probs = F.softmax(classification_logits, dim=-1)
        classification_preds = torch.argmax(classification_probs, dim=-1)
        
        # Add classification results to each prediction instance
        for i, pred in enumerate(preds):
            pred.pred_classification = classification_preds[i].item()
            pred.pred_classification_probs = classification_probs[i].detach().cpu().numpy()
            pred.pred_classification_logits = classification_logits[i].detach().cpu().numpy()
        
        return preds
    
    def loss(self,
             feats: Tuple[torch.Tensor],
             batch_data_samples,
             train_cfg: Optional[dict] = None) -> dict:
        """Calculate losses.
        
        Args:
            feats (Tuple[Tensor]): Multi-level features from backbone
            batch_data_samples: Batch of data samples with labels
            train_cfg (dict, optional): Training configuration
            
        Returns:
            dict: Dictionary of losses
        """
        # Get heatmaps and classification logits
        heatmaps, classification_logits = self.forward(feats)
        
        # Calculate heatmap loss using parent class
        keypoint_losses = super().loss(feats, batch_data_samples, train_cfg)
        
        # Extract ground truth classifications from batch_data_samples
        gt_classifications = []
        for data_sample in batch_data_samples:
            # The ground truth classification should be in the data sample
            # We'll compute it from ground truth landmarks if not provided
            if hasattr(data_sample, 'gt_classification'):
                gt_class = data_sample.gt_classification
            else:
                # Compute from ground truth landmarks
                import sys
                sys.path.insert(0, '.')
                from anb_classification_utils import calculate_anb_angle, classify_from_anb_angle
                
                gt_keypoints = data_sample.gt_instances.keypoints  # Shape: (1, 19, 2)
                anb_angle = calculate_anb_angle(gt_keypoints)
                gt_class = classify_from_anb_angle(anb_angle)
                if isinstance(gt_class, torch.Tensor):
                    gt_class = gt_class.item()
                
            gt_classifications.append(gt_class)
        
        gt_classifications = torch.tensor(gt_classifications, dtype=torch.long, device=classification_logits.device)
        
        # Calculate classification loss
        classification_loss = self.classification_loss(classification_logits, gt_classifications)
        
        # Combine losses
        losses = keypoint_losses.copy()
        losses['loss_classification'] = classification_loss * self.classification_loss_weight
        
        return losses 