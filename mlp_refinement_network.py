#!/usr/bin/env python3
"""
MLP Refinement Network for Cephalometric Landmark Detection
This network refines the predictions from HRNetV2 by using both the initial predictions
and image features to output more accurate landmark coordinates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import cephalometric_dataset_info


class ImageFeatureExtractor(nn.Module):
    """Extract features from the input image using a lightweight CNN."""
    
    def __init__(self, input_size: int = 384, feature_dim: int = 512):
        super().__init__()
        
        # Lightweight feature extractor
        self.features = nn.Sequential(
            # First block: 384x384 -> 192x192
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Second block: 192x192 -> 96x96
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Third block: 96x96 -> 48x48
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Fourth block: 48x48 -> 24x24
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Fifth block: 24x24 -> 12x12
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Global average pooling: 12x12 -> 1x1
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            # Final projection
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor (B, 3, H, W)
        Returns:
            Global image features (B, feature_dim)
        """
        return self.features(x)


class LandmarkRefiner(nn.Module):
    """
    Individual MLP for refining a single landmark.
    Takes initial prediction, local image features, and global context.
    """
    
    def __init__(self, input_dim: int = 4, hidden_dims: list = [256, 128, 64], 
                 dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Final output layer for (x, y) coordinates
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize final layer with small weights for residual learning
        nn.init.normal_(self.network[-1].weight, 0, 0.01)
        nn.init.constant_(self.network[-1].bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, input_dim)
        Returns:
            Coordinate offset (B, 2) - added to initial prediction
        """
        return self.network(x)


class CephalometricMLPRefinement(nn.Module):
    """
    Complete MLP refinement network for cephalometric landmarks.
    
    Architecture:
    1. Extract global image features using lightweight CNN
    2. For each landmark, create input features combining:
       - Initial HRNetV2 prediction (x, y)
       - Distance from image center
       - Global image features
    3. Use individual MLPs per landmark for refinement
    4. Output refined coordinates as residual + initial prediction
    """
    
    def __init__(self, 
                 num_landmarks: int = 19,
                 input_size: int = 384,
                 image_feature_dim: int = 512,
                 landmark_hidden_dims: list = [256, 128, 64],
                 dropout: float = 0.3,
                 use_landmark_weights: bool = True):
        super().__init__()
        
        self.num_landmarks = num_landmarks
        self.input_size = input_size
        self.image_feature_dim = image_feature_dim
        
        # Image feature extractor
        self.image_encoder = ImageFeatureExtractor(input_size, image_feature_dim)
        
        # Input dimension for each landmark refiner:
        # 2 (initial x,y) + 2 (normalized x,y) + 1 (distance from center) + image_feature_dim
        landmark_input_dim = 2 + 2 + 1 + image_feature_dim
        
        # Individual refiners for each landmark
        self.landmark_refiners = nn.ModuleList([
            LandmarkRefiner(
                input_dim=landmark_input_dim,
                hidden_dims=landmark_hidden_dims,
                dropout=dropout
            ) for _ in range(num_landmarks)
        ])
        
        # Landmark weights for focusing on difficult landmarks
        if use_landmark_weights:
            landmark_weights = torch.tensor(
                cephalometric_dataset_info.dataset_info['joint_weights'], 
                dtype=torch.float32
            )
        else:
            landmark_weights = torch.ones(num_landmarks, dtype=torch.float32)
        
        self.register_buffer('landmark_weights', landmark_weights)
        
        # Landmark names for debugging
        self.landmark_names = cephalometric_dataset_info.landmark_names_in_order
    
    def create_landmark_features(self, 
                                initial_predictions: torch.Tensor, 
                                image_features: torch.Tensor) -> torch.Tensor:
        """
        Create input features for each landmark refiner.
        
        Args:
            initial_predictions: (B, num_landmarks, 2) - HRNetV2 predictions
            image_features: (B, image_feature_dim) - Global image features
            
        Returns:
            landmark_features: (B, num_landmarks, feature_dim)
        """
        batch_size = initial_predictions.size(0)
        device = initial_predictions.device
        
        # Normalize coordinates to [-1, 1]
        normalized_preds = initial_predictions / (self.input_size / 2) - 1.0
        
        # Calculate distance from image center
        center = torch.tensor([self.input_size/2, self.input_size/2], 
                             device=device, dtype=torch.float32)
        distances = torch.norm(initial_predictions - center.unsqueeze(0).unsqueeze(0), 
                              dim=-1, keepdim=True)  # (B, num_landmarks, 1)
        
        # Normalize distances
        max_distance = torch.sqrt(torch.tensor(2.0)) * (self.input_size / 2)
        distances = distances / max_distance
        
        # Expand image features for each landmark
        image_features_expanded = image_features.unsqueeze(1).expand(
            batch_size, self.num_landmarks, self.image_feature_dim
        )  # (B, num_landmarks, image_feature_dim)
        
        # Concatenate all features
        landmark_features = torch.cat([
            initial_predictions,      # (B, num_landmarks, 2)
            normalized_preds,         # (B, num_landmarks, 2) 
            distances,                # (B, num_landmarks, 1)
            image_features_expanded   # (B, num_landmarks, image_feature_dim)
        ], dim=-1)  # (B, num_landmarks, 2+2+1+image_feature_dim)
        
        return landmark_features
    
    def forward(self, 
                images: torch.Tensor, 
                initial_predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the refinement network.
        
        Args:
            images: (B, 3, H, W) - Input images
            initial_predictions: (B, num_landmarks, 2) - HRNetV2 predictions
            
        Returns:
            Dictionary containing:
                - 'refined_predictions': (B, num_landmarks, 2) - Final refined coordinates
                - 'refinements': (B, num_landmarks, 2) - Coordinate offsets
                - 'initial_predictions': (B, num_landmarks, 2) - Input predictions
        """
        batch_size = images.size(0)
        
        # Extract global image features
        image_features = self.image_encoder(images)  # (B, image_feature_dim)
        
        # Create input features for landmark refiners
        landmark_features = self.create_landmark_features(
            initial_predictions, image_features
        )  # (B, num_landmarks, feature_dim)
        
        # Refine each landmark independently
        refinements = []
        for i in range(self.num_landmarks):
            landmark_feat = landmark_features[:, i, :]  # (B, feature_dim)
            refinement = self.landmark_refiners[i](landmark_feat)  # (B, 2)
            refinements.append(refinement)
        
        refinements = torch.stack(refinements, dim=1)  # (B, num_landmarks, 2)
        
        # Apply refinements as residual connection
        refined_predictions = initial_predictions + refinements
        
        # Clamp to valid image coordinates
        refined_predictions = torch.clamp(refined_predictions, 0, self.input_size)
        
        return {
            'refined_predictions': refined_predictions,
            'refinements': refinements,
            'initial_predictions': initial_predictions,
            'image_features': image_features
        }
    
    def compute_loss(self, 
                     predictions: Dict[str, torch.Tensor], 
                     targets: torch.Tensor,
                     loss_type: str = 'mse') -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            predictions: Output from forward pass
            targets: (B, num_landmarks, 2) - Ground truth coordinates
            loss_type: 'mse', 'smooth_l1', or 'huber'
            
        Returns:
            Dictionary of losses
        """
        refined_preds = predictions['refined_predictions']
        initial_preds = predictions['initial_predictions']
        
        # Create mask for valid landmarks (non-zero targets)
        valid_mask = (targets[:, :, 0] > 0) & (targets[:, :, 1] > 0)  # (B, num_landmarks)
        
        if valid_mask.sum() == 0:
            # No valid landmarks
            return {
                'total_loss': torch.tensor(0.0, device=targets.device, requires_grad=True),
                'refinement_loss': torch.tensor(0.0, device=targets.device),
                'initial_loss': torch.tensor(0.0, device=targets.device)
            }
        
        # Compute distance errors
        refined_errors = torch.norm(refined_preds - targets, dim=-1)  # (B, num_landmarks)
        initial_errors = torch.norm(initial_preds - targets, dim=-1)  # (B, num_landmarks)
        
        # Apply landmark weights and valid mask
        weights = self.landmark_weights.unsqueeze(0).expand_as(valid_mask)  # (B, num_landmarks)
        weighted_mask = valid_mask.float() * weights
        
        # Compute losses based on type
        if loss_type == 'mse':
            refined_loss = (refined_errors ** 2 * weighted_mask).sum() / weighted_mask.sum()
            initial_loss = (initial_errors ** 2 * weighted_mask).sum() / weighted_mask.sum()
        elif loss_type == 'smooth_l1':
            refined_loss = (F.smooth_l1_loss(refined_preds, targets, reduction='none').mean(dim=-1) * weighted_mask).sum() / weighted_mask.sum()
            initial_loss = (F.smooth_l1_loss(initial_preds, targets, reduction='none').mean(dim=-1) * weighted_mask).sum() / weighted_mask.sum()
        elif loss_type == 'huber':
            refined_loss = (F.huber_loss(refined_preds, targets, reduction='none').mean(dim=-1) * weighted_mask).sum() / weighted_mask.sum()
            initial_loss = (F.huber_loss(initial_preds, targets, reduction='none').mean(dim=-1) * weighted_mask).sum() / weighted_mask.sum()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        total_loss = refined_loss
        
        return {
            'total_loss': total_loss,
            'refinement_loss': refined_loss,
            'initial_loss': initial_loss.detach()  # For monitoring only
        }


def create_model(config: Optional[Dict] = None) -> CephalometricMLPRefinement:
    """
    Factory function to create the MLP refinement model.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured model instance
    """
    if config is None:
        config = {}
    
    return CephalometricMLPRefinement(
        num_landmarks=config.get('num_landmarks', 19),
        input_size=config.get('input_size', 384),
        image_feature_dim=config.get('image_feature_dim', 512),
        landmark_hidden_dims=config.get('landmark_hidden_dims', [256, 128, 64]),
        dropout=config.get('dropout', 0.3),
        use_landmark_weights=config.get('use_landmark_weights', True)
    )


if __name__ == "__main__":
    # Test the network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_model().to(device)
    
    # Test input
    batch_size = 4
    images = torch.randn(batch_size, 3, 384, 384).to(device)
    initial_preds = torch.randn(batch_size, 19, 2).to(device) * 384
    targets = torch.randn(batch_size, 19, 2).to(device) * 384
    
    print("Testing MLP Refinement Network...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(images, initial_preds)
        losses = model.compute_loss(outputs, targets)
    
    print(f"Input shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Initial predictions: {initial_preds.shape}")
    print(f"  Targets: {targets.shape}")
    
    print(f"Output shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    print(f"Losses:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    print("✓ Model test completed successfully!") 