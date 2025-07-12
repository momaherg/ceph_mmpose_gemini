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
        # Extract HeatmapHead specific parameters
        # Set default conv parameters if not provided
        if 'conv_out_channels' not in kwargs:
            kwargs['conv_out_channels'] = (kwargs.get('in_channels', 270),)
        if 'conv_kernel_sizes' not in kwargs:
            kwargs['conv_kernel_sizes'] = (1,)
        
        super().__init__(**kwargs)
        
        self.num_classes = num_classes
        self.classification_loss_weight = classification_loss_weight
        
        # Build classification head
        # We'll use the same input channels as the heatmap head
        in_channels = kwargs.get('in_channels', 270)
        
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
        
    def _process_features(self, feats):
        """Process features to ensure they are in the correct format.
        
        Args:
            feats: Either a tensor or a list/tuple of tensors
            
        Returns:
            Processed features suitable for the head
        """
        # If feats is already a tensor, return as is
        if isinstance(feats, torch.Tensor):
            assert feats.dim() == 4, f"Expected 4D tensor (B,C,H,W), got {feats.dim()}D with shape {feats.shape}"
            return feats
            
        # If feats is a list/tuple, we need to handle it
        if isinstance(feats, (list, tuple)):
            # If it's a single-element list, unwrap it
            if len(feats) == 1:
                feat = feats[0]
                assert feat.dim() == 4, f"Expected 4D tensor (B,C,H,W), got {feat.dim()}D with shape {feat.shape}"
                return feat
            
            # For HRNet multi-scale features, concatenate them like the neck would
            # This handles the case where the neck isn't being called during inference
            if len(feats) == 4:  # HRNet typically outputs 4 scales
                # Ensure all features are 4D
                for i, feat in enumerate(feats):
                    assert feat.dim() == 4, f"Feature {i} has {feat.dim()}D shape {feat.shape}, expected 4D"
                
                # Upsample all features to the same size as the first one
                target_size = feats[0].shape[2:]  # (H, W)
                upsampled_feats = []
                
                for feat in feats:
                    if feat.shape[2:] != target_size:
                        # Upsample to target size
                        feat_up = F.interpolate(
                            feat, 
                            size=target_size, 
                            mode='bilinear', 
                            align_corners=False
                        )
                        upsampled_feats.append(feat_up)
                    else:
                        upsampled_feats.append(feat)
                
                # Concatenate along channel dimension
                concatenated = torch.cat(upsampled_feats, dim=1)
                
                # Verify the result is 4D
                assert concatenated.dim() == 4, f"Concatenated features have {concatenated.dim()}D shape {concatenated.shape}"
                
                # Log that we had to do this
                import mmengine
                try:
                    logger = mmengine.logging.MMLogger.get_current_instance()
                    logger.warning('[HRNetV2WithClassification] Manually concatenated 4 feature maps. '
                                  f'Feature shapes: {[f.shape for f in feats]} -> {concatenated.shape}')
                except:
                    pass
                    
                return concatenated
            
            # For other cases, log a warning and use the last feature map
            import mmengine
            try:
                logger = mmengine.logging.MMLogger.get_current_instance()
                logger.warning(f'[HRNetV2WithClassification] Received {len(feats)} feature maps. '
                              'Expected concatenated features from neck. Using last feature map.')
            except:
                pass
            feat = feats[-1]
            assert feat.dim() == 4, f"Last feature has {feat.dim()}D shape {feat.shape}, expected 4D"
            return feat
        
        raise TypeError(f"Expected tensor or list/tuple of tensors, got {type(feats)}")
    
    def _forward(self, feats):
        """Override parent's _forward to handle multi-scale features.
        
        This method is called by both forward() and predict() in the parent class.
        """
        # Process features to ensure correct format
        processed_feats = self._process_features(feats)
        
        # Call parent's _forward with processed features
        if hasattr(super(), '_forward'):
            return super()._forward(processed_feats)
        else:
            # Fallback: manually implement the forward logic
            x = processed_feats
            
            # Apply conv layers if they exist
            if hasattr(self, 'conv_layers') and self.conv_layers:
                x = self.conv_layers(x)
            
            # Apply deconv layers if they exist
            if hasattr(self, 'deconv_layers') and self.deconv_layers:
                x = self.deconv_layers(x)
                
            # Apply final layer if it exists
            if hasattr(self, 'final_layer') and self.final_layer:
                x = self.final_layer(x)
                
            return x
    
    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """Forward function.
        
        Args:
            feats (Tuple[Tensor]): Multi-level features from backbone
            
        Returns:
            Tensor: heatmaps with shape (N, K, H, W) for K keypoints
            
        Note: This method only returns heatmaps to maintain compatibility with parent class.
        Use forward_with_classification() to get both heatmaps and classification logits.
        """
        # Debug logging
        import mmengine
        try:
            logger = mmengine.logging.MMLogger.get_current_instance()
            if isinstance(feats, (list, tuple)):
                logger.info(f'[HRNetV2WithClassification.forward] Received {len(feats)} features with shapes: {[f.shape for f in feats]}')
            else:
                logger.info(f'[HRNetV2WithClassification.forward] Received tensor with shape: {feats.shape}')
        except:
            pass
            
        # Handle multi-scale features
        if isinstance(feats, (list, tuple)) and not isinstance(feats, torch.Tensor):
            # The parent expects concatenated features from the neck
            # If we're getting raw multi-scale features, process them first
            processed_feats = self._process_features(feats)
            try:
                logger.info(f'[HRNetV2WithClassification.forward] Processed features shape: {processed_feats.shape}')
            except:
                pass
            heatmaps = super().forward(processed_feats)
        else:
            # Already processed by neck
            heatmaps = super().forward(feats)
        return heatmaps
    
    def forward_with_classification(self, feats: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function that returns both heatmaps and classification logits.
        
        Args:
            feats (Tuple[Tensor]): Multi-level features from backbone
            
        Returns:
            Tuple[Tensor, Tensor]: (heatmaps, classification_logits)
                - heatmaps: shape (N, K, H, W) for K keypoints
                - classification_logits: shape (N, num_classes)
        """
        # Get heatmaps from parent class - pass raw features
        heatmaps = super().forward(feats)
        
        # Process features for classification separately
        processed_feats = self._process_features(feats)
        
        # Pass through classification head
        classification_logits = self.classification_head(processed_feats)
        
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
        # Process features to ensure correct format
        processed_feats = self._process_features(feats)
        
        # Get heatmaps and classification logits
        # We can't use forward_with_classification here because super().predict expects raw feats
        # So we need to compute classification separately
        
        # Decode heatmaps to keypoints using parent class method
        # The parent's predict method will call forward internally, so pass raw feats
        preds = super().predict(feats, batch_data_samples, test_cfg)
        
        # Now compute classification logits separately
        with torch.no_grad():
            classification_logits = self.classification_head(processed_feats)
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
        # Get heatmaps and classification logits separately
        heatmaps, classification_logits = self.forward_with_classification(feats)
        
        # Calculate heatmap loss using parent class method
        # We need to call the parent's loss method but pass only heatmaps
        keypoint_losses = super().loss(feats, batch_data_samples, train_cfg)
        
        # Extract ground truth classifications from batch_data_samples
        gt_classifications = []
        for data_sample in batch_data_samples:
            # The ground truth classification should be in the data sample metainfo
            if hasattr(data_sample, 'metainfo') and 'gt_classification' in data_sample.metainfo:
                gt_class = data_sample.metainfo['gt_classification']
            elif hasattr(data_sample, 'gt_classification'):
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