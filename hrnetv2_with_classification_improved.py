"""
Improved HRNetV2 model with classification head that addresses common training issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpose.registry import MODELS
from mmpose.models.heads import HeatmapHead
from typing import Optional, Tuple, Dict


@MODELS.register_module()
class HRNetV2WithClassificationImproved(HeatmapHead):
    """Improved HRNetV2 head with better classification handling.
    
    Key improvements:
    1. Feature adaptation layer for classification
    2. Balanced class weights support
    3. Better initialization
    4. Auxiliary classification loss from intermediate features
    5. Gradient monitoring
    """
    
    def __init__(self,
                 num_classes: int = 3,
                 classification_hidden_dim: int = 256,
                 classification_dropout: float = 0.2,
                 classification_loss_weight: float = 2.0,  # Increased weight
                 class_weights: Optional[list] = None,  # For balanced training
                 use_feature_adapter: bool = True,
                 auxiliary_loss_weight: float = 0.3,
                 **kwargs):
        
        # Initialize parent HeatmapHead
        super().__init__(**kwargs)
        
        self.num_classes = num_classes
        self.classification_loss_weight = classification_loss_weight
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.use_feature_adapter = use_feature_adapter
        
        # Get input channels from parent initialization
        in_channels = self.in_channels
        
        # Feature adaptation layer (optional but recommended)
        if self.use_feature_adapter:
            self.feature_adapter = nn.Sequential(
                nn.Conv2d(in_channels, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
            adapted_channels = 128
        else:
            self.feature_adapter = nn.Identity()
            adapted_channels = in_channels
        
        # Main classification head
        self.classification_head = nn.Sequential(
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # First FC block
            nn.Linear(adapted_channels, classification_hidden_dim),
            nn.BatchNorm1d(classification_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(classification_dropout),
            # Second FC block
            nn.Linear(classification_hidden_dim, classification_hidden_dim // 2),
            nn.BatchNorm1d(classification_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(classification_dropout),
            # Output layer
            nn.Linear(classification_hidden_dim // 2, num_classes)
        )
        
        # Auxiliary classification head (uses features before adaptation)
        self.auxiliary_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )
        
        # Classification loss with optional class weights
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
            self.classification_loss = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.classification_loss = nn.CrossEntropyLoss()
        
        # Initialize classification layers properly
        self._init_classification_weights()
        
        # For gradient monitoring
        self.last_classification_loss = 0.0
        self.last_keypoint_loss = 0.0
    
    def _init_classification_weights(self):
        """Initialize classification layers with proper weights."""
        # Initialize feature adapter
        if self.use_feature_adapter:
            for m in self.feature_adapter.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
        # Initialize classification head
        for m in self.classification_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize auxiliary head
        for m in self.auxiliary_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def loss(self,
             feats: Tuple[torch.Tensor],
             batch_data_samples,
             train_cfg: Optional[dict] = None) -> dict:
        """Calculate losses with improved classification handling."""
        # Get keypoint losses from parent
        losses = super().loss(feats, batch_data_samples, train_cfg)
        self.last_keypoint_loss = sum(v.item() for k, v in losses.items() if 'loss' in k)
        
        # Extract feature tensor
        feat_for_classification = self._extract_feature_tensor(feats)
        if feat_for_classification is None:
            return losses
        
        # Apply feature adaptation
        adapted_features = self.feature_adapter(feat_for_classification)
        
        # Compute main classification logits
        classification_logits = self.classification_head(adapted_features)
        
        # Compute auxiliary classification logits
        auxiliary_logits = self.auxiliary_head(feat_for_classification)
        
        # Extract ground truth classifications
        gt_classifications = self._extract_gt_classifications(batch_data_samples)
        if gt_classifications is None:
            return losses
        
        gt_classifications = torch.tensor(
            gt_classifications, 
            dtype=torch.long, 
            device=classification_logits.device
        )
        
        # Calculate main classification loss
        main_cls_loss = self.classification_loss(classification_logits, gt_classifications)
        
        # Calculate auxiliary classification loss
        aux_cls_loss = self.classification_loss(auxiliary_logits, gt_classifications)
        
        # Combined classification loss
        total_cls_loss = main_cls_loss + self.auxiliary_loss_weight * aux_cls_loss
        
        # Store for monitoring
        self.last_classification_loss = total_cls_loss.item()
        
        # Add to losses
        losses['loss_classification'] = total_cls_loss * self.classification_loss_weight
        losses['loss_classification_main'] = main_cls_loss  # For monitoring
        losses['loss_classification_aux'] = aux_cls_loss    # For monitoring
        
        # Add loss ratio for monitoring
        if self.last_keypoint_loss > 0:
            loss_ratio = self.last_classification_loss / self.last_keypoint_loss
            losses['loss_ratio_cls_kpt'] = torch.tensor(loss_ratio)
        
        return losses
    
    def predict(self,
                feats: Tuple[torch.Tensor],
                batch_data_samples,
                test_cfg: Optional[dict] = None):
        """Predict with improved classification."""
        # Get keypoint predictions from parent
        preds = super().predict(feats, batch_data_samples, test_cfg)
        
        # Extract feature tensor
        feat_for_classification = self._extract_feature_tensor(feats)
        if feat_for_classification is None:
            return preds
        
        # Apply feature adaptation
        adapted_features = self.feature_adapter(feat_for_classification)
        
        # Compute classification
        with torch.no_grad():
            classification_logits = self.classification_head(adapted_features)
            classification_probs = F.softmax(classification_logits, dim=-1)
            classification_preds = torch.argmax(classification_probs, dim=-1)
            
            # Also get auxiliary predictions for analysis
            auxiliary_logits = self.auxiliary_head(feat_for_classification)
            auxiliary_probs = F.softmax(auxiliary_logits, dim=-1)
        
        # Add classification results to predictions
        for i, pred in enumerate(preds):
            # Main predictions
            pred.pred_classification = classification_preds[i:i+1].detach().cpu().numpy()
            pred.pred_classification_probs = classification_probs[i:i+1].detach().cpu().numpy()
            
            # Auxiliary predictions for analysis
            pred.auxiliary_classification_probs = auxiliary_probs[i:i+1].detach().cpu().numpy()
            
            # Also add to pred_instances
            if hasattr(pred, 'pred_instances'):
                pred.pred_instances.pred_classification = classification_preds[i:i+1].detach().cpu().numpy()
                pred.pred_instances.classification_scores = classification_probs[i:i+1].detach().cpu().numpy()
        
        return preds
    
    def _extract_feature_tensor(self, feats):
        """Extract appropriate feature tensor for classification."""
        if isinstance(feats, (list, tuple)):
            if len(feats) == 1:
                feat_for_classification = feats[0]
            else:
                feat_for_classification = feats[-1]
                
            if isinstance(feat_for_classification, (list, tuple)):
                feat_for_classification = feat_for_classification[0]
        else:
            feat_for_classification = feats
        
        if not isinstance(feat_for_classification, torch.Tensor):
            import mmengine
            try:
                logger = mmengine.logging.MMLogger.get_current_instance()
                logger.warning(f'[HRNetV2WithClassificationImproved] Expected tensor, got {type(feat_for_classification)}')
            except:
                pass
            return None
        
        return feat_for_classification
    
    def _extract_gt_classifications(self, batch_data_samples):
        """Extract ground truth classifications from batch."""
        gt_classifications = []
        
        for data_sample in batch_data_samples:
            # Try to get from metainfo first
            if hasattr(data_sample, 'metainfo') and 'gt_classification' in data_sample.metainfo:
                gt_class = data_sample.metainfo['gt_classification']
            else:
                # Compute from ground truth landmarks
                try:
                    import sys
                    sys.path.insert(0, '.')
                    from anb_classification_utils import calculate_anb_angle, classify_from_anb_angle
                    
                    gt_keypoints = data_sample.gt_instances.keypoints
                    anb_angle = calculate_anb_angle(gt_keypoints)
                    gt_class = classify_from_anb_angle(anb_angle)
                    if isinstance(gt_class, torch.Tensor):
                        gt_class = gt_class.item()
                except Exception as e:
                    # Log error and return None
                    import mmengine
                    try:
                        logger = mmengine.logging.MMLogger.get_current_instance()
                        logger.warning(f'[HRNetV2WithClassificationImproved] Failed to extract GT class: {e}')
                    except:
                        pass
                    return None
                    
            gt_classifications.append(gt_class)
        
        return gt_classifications 