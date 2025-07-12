"""
Simplified HRNetV2 model with additional classification head for skeletal pattern prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpose.registry import MODELS
from mmpose.models.heads import HeatmapHead
from typing import Optional, Tuple


@MODELS.register_module()
class HRNetV2WithClassificationSimple(HeatmapHead):
    """Simplified HRNetV2 head with additional classification output.
    
    This version minimally extends HeatmapHead to add classification without
    interfering with the parent's forward logic.
    """
    
    def __init__(self,
                 num_classes: int = 3,
                 classification_hidden_dim: int = 256,
                 classification_dropout: float = 0.2,
                 classification_loss_weight: float = 0.5,
                 **kwargs):
        
        # Initialize parent HeatmapHead
        super().__init__(**kwargs)
        
        self.num_classes = num_classes
        self.classification_loss_weight = classification_loss_weight
        
        # Build classification head
        # Get input channels from parent initialization
        in_channels = self.in_channels
        
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
        
        # Classification loss
        self.classification_loss = nn.CrossEntropyLoss()
    
    def loss(self,
             feats: Tuple[torch.Tensor],
             batch_data_samples,
             train_cfg: Optional[dict] = None) -> dict:
        """Calculate losses.
        
        Args:
            feats: Features from backbone/neck
            batch_data_samples: Batch of data samples with labels
            train_cfg: Training configuration
            
        Returns:
            dict: Dictionary of losses
        """
        # Get keypoint losses from parent
        losses = super().loss(feats, batch_data_samples, train_cfg)
        
        # Add classification loss
        # Extract the feature tensor for classification
        if isinstance(feats, (list, tuple)):
            # Make sure we get a tensor, not a nested list
            if len(feats) == 1:
                feat_for_classification = feats[0]
            else:
                # Use the last feature map
                feat_for_classification = feats[-1]
                
            # If it's still a list, try to extract the tensor
            if isinstance(feat_for_classification, (list, tuple)):
                feat_for_classification = feat_for_classification[0]
        else:
            feat_for_classification = feats
            
        # Ensure we have a tensor
        if not isinstance(feat_for_classification, torch.Tensor):
            # Log warning and return without classification loss
            import mmengine
            try:
                logger = mmengine.logging.MMLogger.get_current_instance()
                logger.warning(f'[HRNetV2WithClassificationSimple] Expected tensor for classification, got {type(feat_for_classification)}. Skipping classification loss.')
            except:
                pass
            return losses
            
        # Compute classification logits
        classification_logits = self.classification_head(feat_for_classification)
        
        # Extract ground truth classifications
        gt_classifications = []
        for data_sample in batch_data_samples:
            # Try to get from metainfo first
            if hasattr(data_sample, 'metainfo') and 'gt_classification' in data_sample.metainfo:
                gt_class = data_sample.metainfo['gt_classification']
            else:
                # Compute from ground truth landmarks
                import sys
                sys.path.insert(0, '.')
                from anb_classification_utils import calculate_anb_angle, classify_from_anb_angle
                
                gt_keypoints = data_sample.gt_instances.keypoints
                anb_angle = calculate_anb_angle(gt_keypoints)
                gt_class = classify_from_anb_angle(anb_angle)
                if isinstance(gt_class, torch.Tensor):
                    gt_class = gt_class.item()
                    
            gt_classifications.append(gt_class)
        
        gt_classifications = torch.tensor(
            gt_classifications, 
            dtype=torch.long, 
            device=classification_logits.device
        )
        
        # Calculate classification loss
        classification_loss = self.classification_loss(
            classification_logits, 
            gt_classifications
        )
        
        # Add to losses
        losses['loss_classification'] = classification_loss * self.classification_loss_weight
        
        return losses
    
    def predict(self,
                feats: Tuple[torch.Tensor],
                batch_data_samples,
                test_cfg: Optional[dict] = None):
        """Predict function for inference.
        
        Args:
            feats: Features from backbone/neck
            batch_data_samples: Batch of data samples
            test_cfg: Test configuration
            
        Returns:
            Predictions with both keypoints and classification
        """
        # Get keypoint predictions from parent
        preds = super().predict(feats, batch_data_samples, test_cfg)
        
        # Add classification predictions
        # Extract the feature tensor for classification
        if isinstance(feats, (list, tuple)):
            # Make sure we get a tensor, not a nested list
            if len(feats) == 1:
                feat_for_classification = feats[0]
            else:
                # Use the last feature map
                feat_for_classification = feats[-1]
                
            # If it's still a list, try to extract the tensor
            if isinstance(feat_for_classification, (list, tuple)):
                feat_for_classification = feat_for_classification[0]
        else:
            feat_for_classification = feats
            
        # Ensure we have a tensor
        if not isinstance(feat_for_classification, torch.Tensor):
            # Log warning and skip classification
            import mmengine
            try:
                logger = mmengine.logging.MMLogger.get_current_instance()
                logger.warning(f'[HRNetV2WithClassificationSimple] Expected tensor for classification, got {type(feat_for_classification)}. Skipping classification.')
            except:
                pass
            # Return predictions without classification
            return preds
            
        # Compute classification
        with torch.no_grad():
            classification_logits = self.classification_head(feat_for_classification)
            classification_probs = F.softmax(classification_logits, dim=-1)
            classification_preds = torch.argmax(classification_probs, dim=-1)
        
        # Add classification results to predictions
        for i, pred in enumerate(preds):
            pred.pred_classification = classification_preds[i].item()
            pred.pred_classification_probs = classification_probs[i].detach().cpu().numpy()
            
        return preds 