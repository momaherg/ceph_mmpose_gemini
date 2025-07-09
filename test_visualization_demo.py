#!/usr/bin/env python3
"""
Demo script to test visualization functionality
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple

def create_demo_data(n_patients=20, n_landmarks=19):
    """Create synthetic data for testing visualization."""
    # Create ground truth
    gt_coords = np.random.rand(n_patients, n_landmarks, 2) * 200 + 12
    
    # Create predictions with some error
    ensemble_hrnet = gt_coords + np.random.randn(n_patients, n_landmarks, 2) * 3
    ensemble_mlp = gt_coords + np.random.randn(n_patients, n_landmarks, 2) * 2  # MLP is slightly better
    
    # Create individual model predictions
    all_hrnet_preds = []
    all_mlp_preds = []
    for i in range(3):
        hrnet_pred = gt_coords + np.random.randn(n_patients, n_landmarks, 2) * 3.5
        mlp_pred = gt_coords + np.random.randn(n_patients, n_landmarks, 2) * 2.5
        all_hrnet_preds.append(hrnet_pred)
        all_mlp_preds.append(mlp_pred)
    
    # Create patient IDs
    patient_ids = list(range(1000, 1000 + n_patients))
    
    # Create landmark names
    landmark_names = [f'landmark_{i}' for i in range(n_landmarks)]
    landmark_names[0] = 'sella'
    landmark_names[1] = 'Gonion'
    landmark_names[2] = 'PNS'
    
    return {
        'gt_coords': gt_coords,
        'ensemble_hrnet': ensemble_hrnet,
        'ensemble_mlp': ensemble_mlp,
        'all_hrnet_preds': all_hrnet_preds,
        'all_mlp_preds': all_mlp_preds,
        'patient_ids': patient_ids,
        'landmark_names': landmark_names
    }

def demo_visualization():
    """Run a simple demo of the visualization functionality."""
    print("🎨 Creating demo visualization...")
    
    # Create demo data
    data = create_demo_data()
    
    # Create output directory
    output_dir = "demo_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a simple visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Error distribution
    ax = axes[0, 0]
    patient_errors = []
    for i in range(len(data['patient_ids'])):
        errors = np.sqrt(np.sum((data['ensemble_mlp'][i] - data['gt_coords'][i])**2, axis=1))
        patient_errors.append(np.mean(errors))
    
    ax.hist(patient_errors, bins=15, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('Average Error per Patient')
    ax.set_ylabel('Count')
    ax.set_title('Patient Error Distribution (Demo)')
    
    # Plot 2: Model comparison
    ax = axes[0, 1]
    hrnet_errors = np.mean([np.mean(np.sqrt(np.sum((p - data['gt_coords'])**2, axis=2))) 
                            for p in data['all_hrnet_preds']])
    mlp_errors = np.mean([np.mean(np.sqrt(np.sum((p - data['gt_coords'])**2, axis=2))) 
                          for p in data['all_mlp_preds']])
    ens_hrnet_error = np.mean(np.sqrt(np.sum((data['ensemble_hrnet'] - data['gt_coords'])**2, axis=2)))
    ens_mlp_error = np.mean(np.sqrt(np.sum((data['ensemble_mlp'] - data['gt_coords'])**2, axis=2)))
    
    models = ['Avg Model\nHRNet', 'Avg Model\nMLP', 'Ensemble\nHRNet', 'Ensemble\nMLP']
    errors = [hrnet_errors, mlp_errors, ens_hrnet_error, ens_mlp_error]
    colors = ['lightblue', 'lightcoral', 'blue', 'red']
    
    bars = ax.bar(models, errors, color=colors, alpha=0.7)
    ax.set_ylabel('Mean Error')
    ax.set_title('Model Performance Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Sample patient visualization
    ax = axes[1, 0]
    patient_idx = 0
    gt = data['gt_coords'][patient_idx]
    pred = data['ensemble_mlp'][patient_idx]
    
    ax.scatter(gt[:, 0], gt[:, 1], c='green', s=50, marker='o', label='Ground Truth', alpha=0.8)
    ax.scatter(pred[:, 0], pred[:, 1], c='red', s=30, marker='s', label='Ensemble MLP', alpha=0.8)
    
    # Draw connections
    for g, p in zip(gt, pred):
        ax.plot([g[0], p[0]], [g[1], p[1]], 'gray', alpha=0.3, linewidth=0.5)
    
    ax.set_xlim(0, 224)
    ax.set_ylim(224, 0)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(f'Sample Patient {data["patient_ids"][patient_idx]} Predictions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Landmark-wise improvement
    ax = axes[1, 1]
    improvements = []
    for j in range(len(data['landmark_names'])):
        hrnet_err = np.mean(np.sqrt(np.sum((data['ensemble_hrnet'][:, j] - data['gt_coords'][:, j])**2, axis=1)))
        mlp_err = np.mean(np.sqrt(np.sum((data['ensemble_mlp'][:, j] - data['gt_coords'][:, j])**2, axis=1)))
        improvement = (hrnet_err - mlp_err) / hrnet_err * 100 if hrnet_err > 0 else 0
        improvements.append(improvement)
    
    # Show top 5 landmarks
    top_indices = np.argsort(improvements)[-5:]
    top_landmarks = [data['landmark_names'][i] for i in top_indices]
    top_improvements = [improvements[i] for i in top_indices]
    
    ax.barh(range(len(top_landmarks)), top_improvements, color='green', alpha=0.7)
    ax.set_yticks(range(len(top_landmarks)))
    ax.set_yticklabels(top_landmarks)
    ax.set_xlabel('Improvement (%)')
    ax.set_title('Top 5 Landmark Improvements')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Visualization Demo - Synthetic Data', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'demo_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Demo visualization saved to: {output_path}")
    
    # Print summary
    print(f"\n📊 Demo Summary:")
    print(f"   - Number of patients: {len(data['patient_ids'])}")
    print(f"   - Number of landmarks: {len(data['landmark_names'])}")
    print(f"   - Average ensemble MLP error: {ens_mlp_error:.2f}")
    print(f"   - Average ensemble HRNet error: {ens_hrnet_error:.2f}")
    print(f"   - Improvement: {(ens_hrnet_error - ens_mlp_error) / ens_hrnet_error * 100:.1f}%")

if __name__ == "__main__":
    demo_visualization() 