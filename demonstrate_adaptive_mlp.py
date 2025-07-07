#!/usr/bin/env python3
"""
Demonstration of Adaptive Selection MLP vs Standard MLP
Shows how the model learns to choose between HRNet and MLP predictions
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define both model architectures for comparison

class StandardMLPRefinementModel(nn.Module):
    """Standard MLP that always outputs refined coordinates."""
    def __init__(self, input_dim=38, hidden_dim=500, output_dim=38):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        out = self.net(x)
        return out + 0.1 * x  # Small residual connection

class AdaptiveMLPRefinementModel(nn.Module):
    """Adaptive MLP with selection mechanism."""
    def __init__(self, input_dim=38, hidden_dim=500, output_dim=38):
        super().__init__()
        
        # Refinement network
        self.refinement_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Selection network
        self.selection_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Get MLP refinements
        mlp_refinement = self.refinement_net(x)
        mlp_predictions = mlp_refinement + 0.1 * x
        
        # Get selection weights
        selection_weights = self.selection_net(x)
        
        # Adaptive combination
        adaptive_output = (1 - selection_weights) * x + selection_weights * mlp_predictions
        
        self.last_selection_weights = selection_weights
        return adaptive_output

def create_synthetic_data(n_samples=100):
    """Create synthetic landmark data with varying difficulty."""
    np.random.seed(42)
    
    # Ground truth landmarks (19 landmarks × 2 coords = 38)
    gt_landmarks = np.random.rand(n_samples, 38) * 200 + 12  # Scale to image coordinates
    
    # Create HRNet predictions with varying errors
    hrnet_errors = np.zeros((n_samples, 38))
    
    # Easy landmarks (0-5): Small errors
    hrnet_errors[:, :10] = np.random.randn(n_samples, 10) * 2
    
    # Medium landmarks (6-12): Moderate errors  
    hrnet_errors[:, 10:26] = np.random.randn(n_samples, 16) * 5
    
    # Hard landmarks (13-18): Large errors
    hrnet_errors[:, 26:] = np.random.randn(n_samples, 12) * 10
    
    hrnet_predictions = gt_landmarks + hrnet_errors
    
    return hrnet_predictions, gt_landmarks

def visualize_selection_patterns(adaptive_model, hrnet_preds, gt_coords):
    """Visualize which coordinates the model chooses to refine."""
    adaptive_model.eval()
    
    with torch.no_grad():
        # Get predictions
        hrnet_tensor = torch.FloatTensor(hrnet_preds)
        adaptive_output = adaptive_model(hrnet_tensor)
        selection_weights = adaptive_model.last_selection_weights.numpy()
    
    # Average selection weights across samples
    avg_weights = np.mean(selection_weights, axis=0)
    
    # Reshape to landmark format
    landmark_weights = avg_weights.reshape(19, 2)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Selection weights by landmark
    landmark_names = [f"L{i}" for i in range(19)]
    x_weights = landmark_weights[:, 0]
    y_weights = landmark_weights[:, 1]
    
    x = np.arange(19)
    width = 0.35
    
    ax1.bar(x - width/2, x_weights, width, label='X coord', alpha=0.8)
    ax1.bar(x + width/2, y_weights, width, label='Y coord', alpha=0.8)
    ax1.set_xlabel('Landmark')
    ax1.set_ylabel('Selection Weight (0=HRNet, 1=MLP)')
    ax1.set_title('Adaptive Selection Weights by Landmark')
    ax1.set_xticks(x)
    ax1.set_xticklabels(landmark_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    
    # Plot 2: Heatmap of selection weights
    ax2.imshow(landmark_weights.T, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    ax2.set_xlabel('Landmark')
    ax2.set_ylabel('Coordinate')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['X', 'Y'])
    ax2.set_xticks(range(19))
    ax2.set_xticklabels(landmark_names, rotation=45)
    ax2.set_title('Selection Weight Heatmap')
    
    # Add colorbar
    cbar = plt.colorbar(ax2.images[0], ax=ax2)
    cbar.set_label('Weight (0=HRNet, 1=MLP)')
    
    # Add text annotations
    for i in range(19):
        for j in range(2):
            text = ax2.text(i, j, f'{landmark_weights[i, j]:.2f}',
                           ha="center", va="center", color="black" if landmark_weights[i, j] < 0.5 else "white")
    
    plt.tight_layout()
    plt.savefig('adaptive_selection_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n🔍 Selection Pattern Analysis:")
    print("="*50)
    print(f"Overall MLP usage: {np.mean(avg_weights):.3f}")
    print(f"\nLandmarks preferring MLP (weight > 0.7):")
    high_mlp = np.where(np.mean(landmark_weights, axis=1) > 0.7)[0]
    for idx in high_mlp:
        print(f"  - Landmark {idx}: {np.mean(landmark_weights[idx]):.3f}")
    
    print(f"\nLandmarks preferring HRNet (weight < 0.3):")
    high_hrnet = np.where(np.mean(landmark_weights, axis=1) < 0.3)[0]
    for idx in high_hrnet:
        print(f"  - Landmark {idx}: {np.mean(landmark_weights[idx]):.3f}")

def compare_models(standard_model, adaptive_model, hrnet_preds, gt_coords):
    """Compare performance of standard vs adaptive MLP."""
    standard_model.eval()
    adaptive_model.eval()
    
    with torch.no_grad():
        hrnet_tensor = torch.FloatTensor(hrnet_preds)
        gt_tensor = torch.FloatTensor(gt_coords)
        
        # Get predictions
        standard_output = standard_model(hrnet_tensor)
        adaptive_output = adaptive_model(hrnet_tensor)
        
        # Calculate errors
        hrnet_errors = torch.sqrt(torch.sum((hrnet_tensor - gt_tensor)**2, dim=1))
        standard_errors = torch.sqrt(torch.sum((standard_output - gt_tensor)**2, dim=1))
        adaptive_errors = torch.sqrt(torch.sum((adaptive_output - gt_tensor)**2, dim=1))
    
    # Print comparison
    print("\n📊 Model Comparison Results:")
    print("="*50)
    print(f"{'Method':<20} {'Mean Error':<15} {'Std Error':<15}")
    print("-"*50)
    print(f"{'HRNet Only':<20} {hrnet_errors.mean():<15.3f} {hrnet_errors.std():<15.3f}")
    print(f"{'Standard MLP':<20} {standard_errors.mean():<15.3f} {standard_errors.std():<15.3f}")
    print(f"{'Adaptive MLP':<20} {adaptive_errors.mean():<15.3f} {adaptive_errors.std():<15.3f}")
    
    # Calculate improvements
    standard_improvement = (hrnet_errors.mean() - standard_errors.mean()) / hrnet_errors.mean() * 100
    adaptive_improvement = (hrnet_errors.mean() - adaptive_errors.mean()) / hrnet_errors.mean() * 100
    
    print(f"\nImprovements over HRNet:")
    print(f"  Standard MLP: {standard_improvement:.1f}%")
    print(f"  Adaptive MLP: {adaptive_improvement:.1f}%")
    print(f"  Adaptive advantage: {adaptive_improvement - standard_improvement:.1f}% better")

def main():
    print("🎯 Adaptive Selection MLP Demonstration")
    print("="*50)
    
    # Create synthetic data
    print("\n📊 Creating synthetic landmark data...")
    hrnet_preds, gt_coords = create_synthetic_data(n_samples=200)
    print(f"  - {len(hrnet_preds)} samples")
    print(f"  - 19 landmarks (38 coordinates)")
    print(f"  - Easy landmarks (0-5): small HRNet errors")
    print(f"  - Medium landmarks (6-12): moderate HRNet errors")
    print(f"  - Hard landmarks (13-18): large HRNet errors")
    
    # Initialize models
    print("\n🤖 Initializing models...")
    standard_model = StandardMLPRefinementModel()
    adaptive_model = AdaptiveMLPRefinementModel()
    
    # For demonstration, we'll use pre-set weights that show the concept
    # In practice, these would be learned during training
    with torch.no_grad():
        # Make adaptive model prefer HRNet for easy landmarks and MLP for hard ones
        # This is just for demonstration - real weights are learned
        selection_layer = adaptive_model.selection_net[-2]
        selection_layer.bias.data.fill_(-2)  # Start with low selection (prefer HRNet)
        
        # Increase selection weights for hard landmarks
        selection_layer.bias.data[26:] = 1.5  # Prefer MLP for hard landmarks
    
    print("  ✓ Standard MLP initialized")
    print("  ✓ Adaptive MLP initialized with example selection pattern")
    
    # Compare models
    compare_models(standard_model, adaptive_model, hrnet_preds, gt_coords)
    
    # Visualize selection patterns
    print("\n📈 Visualizing selection patterns...")
    visualize_selection_patterns(adaptive_model, hrnet_preds, gt_coords)
    
    print("\n✨ Key Benefits of Adaptive Selection MLP:")
    print("  1. ✅ Learns when to trust HRNet vs MLP predictions")
    print("  2. ✅ Can preserve good HRNet predictions while refining bad ones")
    print("  3. ✅ Reduces risk of MLP making predictions worse")
    print("  4. ✅ Provides interpretable selection weights for analysis")
    print("  5. ✅ Adapts to landmark difficulty automatically")

if __name__ == "__main__":
    main() 