#!/usr/bin/env python3
"""
Evaluation Script for MLP Refinement Network
Evaluates the refinement model and compares with HRNetV2 baseline.
"""

import os
import torch
import pandas as pd
import numpy as np
import warnings
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from mmengine.registry import init_default_scope

# Import custom modules
from mlp_refinement_network import CephalometricMLPRefinement, create_model
from mlp_refinement_dataset import MLPRefinementDataset
import cephalometric_dataset_info

warnings.filterwarnings('ignore')

# Apply PyTorch safe loading fix
import functools
_original_torch_load = torch.load
def safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = safe_torch_load


class MLPRefinementEvaluator:
    """Comprehensive evaluator for MLP refinement network."""
    
    def __init__(self, 
                 model: CephalometricMLPRefinement,
                 dataset: MLPRefinementDataset,
                 device: str = 'cuda:0'):
        """
        Args:
            model: Trained MLP refinement model
            dataset: Evaluation dataset
            device: Evaluation device
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.landmark_names = cephalometric_dataset_info.landmark_names_in_order
        
        # Results storage
        self.results = {
            'hrnet_predictions': [],
            'refined_predictions': [],
            'ground_truth': [],
            'valid_masks': [],
            'improvements': []
        }
    
    def evaluate_sample(self, idx: int) -> Dict[str, np.ndarray]:
        """Evaluate a single sample."""
        sample = self.dataset[idx]
        
        # Move to device
        image = sample['image'].unsqueeze(0).to(self.device)
        hrnet_preds = sample['hrnet_predictions'].unsqueeze(0).to(self.device)
        targets = sample['ground_truth'].unsqueeze(0).to(self.device)
        valid_mask = sample['valid_mask'].unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            predictions = self.model(image, hrnet_preds)
        
        # Convert to numpy
        hrnet_np = hrnet_preds.cpu().numpy()[0]
        refined_np = predictions['refined_predictions'].cpu().numpy()[0]
        targets_np = targets.cpu().numpy()[0]
        valid_np = valid_mask.cpu().numpy()[0]
        
        # Compute errors
        hrnet_errors = np.linalg.norm(hrnet_np - targets_np, axis=1)
        refined_errors = np.linalg.norm(refined_np - targets_np, axis=1)
        improvements = hrnet_errors - refined_errors
        
        return {
            'hrnet_predictions': hrnet_np,
            'refined_predictions': refined_np,
            'ground_truth': targets_np,
            'valid_mask': valid_np,
            'hrnet_errors': hrnet_errors,
            'refined_errors': refined_errors,
            'improvements': improvements
        }
    
    def evaluate_all(self) -> Dict[str, float]:
        """Evaluate all samples in the dataset."""
        print(f"Evaluating {len(self.dataset)} samples...")
        
        all_hrnet_errors = []
        all_refined_errors = []
        all_improvements = []
        per_landmark_hrnet = {name: [] for name in self.landmark_names}
        per_landmark_refined = {name: [] for name in self.landmark_names}
        per_landmark_improvements = {name: [] for name in self.landmark_names}
        
        self.model.eval()
        
        for idx in range(len(self.dataset)):
            if idx % 50 == 0:
                print(f"  Processed {idx}/{len(self.dataset)} samples")
            
            result = self.evaluate_sample(idx)
            
            # Store results
            self.results['hrnet_predictions'].append(result['hrnet_predictions'])
            self.results['refined_predictions'].append(result['refined_predictions'])
            self.results['ground_truth'].append(result['ground_truth'])
            self.results['valid_masks'].append(result['valid_mask'])
            self.results['improvements'].append(result['improvements'])
            
            # Collect valid errors
            valid_mask = result['valid_mask'] > 0
            if valid_mask.sum() > 0:
                valid_hrnet_errors = result['hrnet_errors'][valid_mask]
                valid_refined_errors = result['refined_errors'][valid_mask]
                valid_improvements = result['improvements'][valid_mask]
                
                all_hrnet_errors.extend(valid_hrnet_errors)
                all_refined_errors.extend(valid_refined_errors)
                all_improvements.extend(valid_improvements)
                
                # Per-landmark collection
                for i, (name, is_valid) in enumerate(zip(self.landmark_names, valid_mask)):
                    if is_valid:
                        per_landmark_hrnet[name].append(result['hrnet_errors'][i])
                        per_landmark_refined[name].append(result['refined_errors'][i])
                        per_landmark_improvements[name].append(result['improvements'][i])
        
        # Compute overall metrics
        all_hrnet_errors = np.array(all_hrnet_errors)
        all_refined_errors = np.array(all_refined_errors)
        all_improvements = np.array(all_improvements)
        
        overall_metrics = {
            'hrnet_mre': np.mean(all_hrnet_errors),
            'hrnet_std': np.std(all_hrnet_errors),
            'refined_mre': np.mean(all_refined_errors),
            'refined_std': np.std(all_refined_errors),
            'improvement_mean': np.mean(all_improvements),
            'improvement_std': np.std(all_improvements),
            'improvement_percentage': (1 - np.mean(all_refined_errors) / np.mean(all_hrnet_errors)) * 100
        }
        
        # Per-landmark metrics
        per_landmark_metrics = {}
        for name in self.landmark_names:
            if len(per_landmark_hrnet[name]) > 0:
                hrnet_errors = np.array(per_landmark_hrnet[name])
                refined_errors = np.array(per_landmark_refined[name])
                improvements = np.array(per_landmark_improvements[name])
                
                per_landmark_metrics[name] = {
                    'hrnet_mre': np.mean(hrnet_errors),
                    'hrnet_std': np.std(hrnet_errors),
                    'refined_mre': np.mean(refined_errors),
                    'refined_std': np.std(refined_errors),
                    'improvement_mean': np.mean(improvements),
                    'improvement_percentage': (1 - np.mean(refined_errors) / np.mean(hrnet_errors)) * 100,
                    'count': len(hrnet_errors)
                }
        
        self.overall_metrics = overall_metrics
        self.per_landmark_metrics = per_landmark_metrics
        
        return overall_metrics
    
    def print_results(self):
        """Print evaluation results."""
        print("\n" + "="*80)
        print("MLP REFINEMENT EVALUATION RESULTS")
        print("="*80)
        
        # Overall results
        print(f"\nOVERALL PERFORMANCE:")
        print(f"HRNetV2 Baseline MRE: {self.overall_metrics['hrnet_mre']:.3f} ± {self.overall_metrics['hrnet_std']:.3f} pixels")
        print(f"Refined MRE:           {self.overall_metrics['refined_mre']:.3f} ± {self.overall_metrics['refined_std']:.3f} pixels")
        print(f"Improvement:           {self.overall_metrics['improvement_mean']:+.3f} pixels ({self.overall_metrics['improvement_percentage']:+.1f}%)")
        
        # Percentile analysis
        all_hrnet_errors = []
        all_refined_errors = []
        for result in self.results['improvements']:
            valid_mask = self.results['valid_masks'][len(all_hrnet_errors)] > 0
            if valid_mask.sum() > 0:
                hrnet_errors = np.linalg.norm(
                    self.results['hrnet_predictions'][len(all_hrnet_errors)] - 
                    self.results['ground_truth'][len(all_hrnet_errors)], axis=1
                )[valid_mask]
                refined_errors = np.linalg.norm(
                    self.results['refined_predictions'][len(all_refined_errors)] - 
                    self.results['ground_truth'][len(all_refined_errors)], axis=1
                )[valid_mask]
                all_hrnet_errors.extend(hrnet_errors)
                all_refined_errors.extend(refined_errors)
        
        all_hrnet_errors = np.array(all_hrnet_errors)
        all_refined_errors = np.array(all_refined_errors)
        
        print(f"\nPERCENTILE ANALYSIS:")
        print(f"                   HRNetV2    Refined    Improvement")
        for p in [50, 90, 95]:
            hrnet_p = np.percentile(all_hrnet_errors, p)
            refined_p = np.percentile(all_refined_errors, p)
            improvement_p = hrnet_p - refined_p
            print(f"{p:2d}th percentile:   {hrnet_p:6.3f}    {refined_p:6.3f}    {improvement_p:+6.3f}")
        
        # Per-landmark results
        print(f"\nPER-LANDMARK IMPROVEMENTS:")
        print(f"{'Index':<5} {'Landmark':<20} {'HRNet MRE':<10} {'Refined MRE':<11} {'Improvement':<12} {'%':<8} {'Count':<6}")
        print("-" * 85)
        
        for i, name in enumerate(self.landmark_names):
            if name in self.per_landmark_metrics:
                stats = self.per_landmark_metrics[name]
                print(f"{i:<5} {name:<20} {stats['hrnet_mre']:<10.3f} {stats['refined_mre']:<11.3f} "
                      f"{stats['improvement_mean']:<+12.3f} {stats['improvement_percentage']:<+8.1f} {stats['count']:<6}")
        
        # Highlight challenging landmarks
        print(f"\nCHALLENGING LANDMARKS (>3px error):")
        challenging_landmarks = []
        for name, stats in self.per_landmark_metrics.items():
            if stats['hrnet_mre'] > 3.0:
                challenging_landmarks.append((name, stats))
        
        challenging_landmarks.sort(key=lambda x: x[1]['hrnet_mre'], reverse=True)
        for name, stats in challenging_landmarks[:5]:
            print(f"  {name}: {stats['hrnet_mre']:.3f} → {stats['refined_mre']:.3f} "
                  f"({stats['improvement_percentage']:+.1f}%)")
    
    def plot_results(self, save_dir: str):
        """Create comprehensive result plots."""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Overall error distribution comparison
        self._plot_error_distribution(save_dir)
        
        # 2. Per-landmark comparison
        self._plot_per_landmark_comparison(save_dir)
        
        # 3. Improvement scatter plot
        self._plot_improvement_scatter(save_dir)
        
        # 4. Challenging landmarks focus
        self._plot_challenging_landmarks(save_dir)
        
        print(f"Plots saved to {save_dir}")
    
    def _plot_error_distribution(self, save_dir: str):
        """Plot error distribution comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Collect all errors
        all_hrnet_errors = []
        all_refined_errors = []
        
        for i in range(len(self.results['hrnet_predictions'])):
            valid_mask = self.results['valid_masks'][i] > 0
            if valid_mask.sum() > 0:
                hrnet_errors = np.linalg.norm(
                    self.results['hrnet_predictions'][i] - self.results['ground_truth'][i], axis=1
                )[valid_mask]
                refined_errors = np.linalg.norm(
                    self.results['refined_predictions'][i] - self.results['ground_truth'][i], axis=1
                )[valid_mask]
                
                all_hrnet_errors.extend(hrnet_errors)
                all_refined_errors.extend(refined_errors)
        
        # Error distribution
        ax1.hist(all_hrnet_errors, bins=50, alpha=0.7, label='HRNetV2', color='skyblue', density=True)
        ax1.hist(all_refined_errors, bins=50, alpha=0.7, label='Refined', color='lightcoral', density=True)
        ax1.set_xlabel('Error (pixels)')
        ax1.set_ylabel('Density')
        ax1.set_title('Error Distribution Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot comparison
        ax2.boxplot([all_hrnet_errors, all_refined_errors], 
                   labels=['HRNetV2', 'Refined'],
                   patch_artist=True,
                   boxprops=[dict(facecolor='skyblue'), dict(facecolor='lightcoral')])
        ax2.set_ylabel('Error (pixels)')
        ax2.set_title('Error Distribution Box Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'error_distribution_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_per_landmark_comparison(self, save_dir: str):
        """Plot per-landmark comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        landmark_indices = list(range(len(self.landmark_names)))
        hrnet_mres = [self.per_landmark_metrics[name]['hrnet_mre'] for name in self.landmark_names]
        refined_mres = [self.per_landmark_metrics[name]['refined_mre'] for name in self.landmark_names]
        improvements = [self.per_landmark_metrics[name]['improvement_mean'] for name in self.landmark_names]
        
        # MRE comparison
        x = np.arange(len(landmark_indices))
        width = 0.35
        
        ax1.bar(x - width/2, hrnet_mres, width, label='HRNetV2', color='skyblue')
        ax1.bar(x + width/2, refined_mres, width, label='Refined', color='lightcoral')
        ax1.set_xlabel('Landmark Index')
        ax1.set_ylabel('MRE (pixels)')
        ax1.set_title('Per-Landmark MRE Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(i) for i in landmark_indices])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Improvement plot
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax2.bar(landmark_indices, improvements, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Landmark Index')
        ax2.set_ylabel('Improvement (pixels)')
        ax2.set_title('Per-Landmark Improvement')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, imp) in enumerate(zip(bars, improvements)):
            if abs(imp) > 0.05:  # Only show significant improvements
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{imp:+.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'per_landmark_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_improvement_scatter(self, save_dir: str):
        """Plot improvement scatter plot."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Collect data for scatter plot
        hrnet_errors = []
        improvements = []
        landmark_labels = []
        
        for name in self.landmark_names:
            if name in self.per_landmark_metrics:
                stats = self.per_landmark_metrics[name]
                hrnet_errors.append(stats['hrnet_mre'])
                improvements.append(stats['improvement_mean'])
                landmark_labels.append(name)
        
        # Create scatter plot
        scatter = ax.scatter(hrnet_errors, improvements, 
                           c=improvements, cmap='RdYlGn', 
                           s=100, alpha=0.7, edgecolors='black')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Improvement (pixels)')
        
        # Add reference lines
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.axvline(x=np.mean(hrnet_errors), color='blue', linestyle='--', alpha=0.5, label='Mean HRNet Error')
        
        # Add labels for challenging landmarks
        for i, (x, y, label) in enumerate(zip(hrnet_errors, improvements, landmark_labels)):
            if x > 3.0:  # Challenging landmarks
                ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
        
        ax.set_xlabel('HRNetV2 MRE (pixels)')
        ax.set_ylabel('Improvement (pixels)')
        ax.set_title('Improvement vs. Initial Error')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'improvement_scatter.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_challenging_landmarks(self, save_dir: str):
        """Focus plot on challenging landmarks."""
        challenging_landmarks = []
        for name, stats in self.per_landmark_metrics.items():
            if stats['hrnet_mre'] > 3.0:
                challenging_landmarks.append((name, stats))
        
        if not challenging_landmarks:
            return
        
        challenging_landmarks.sort(key=lambda x: x[1]['hrnet_mre'], reverse=True)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        names = [item[0] for item in challenging_landmarks]
        hrnet_mres = [item[1]['hrnet_mre'] for item in challenging_landmarks]
        refined_mres = [item[1]['refined_mre'] for item in challenging_landmarks]
        
        x = np.arange(len(names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, hrnet_mres, width, label='HRNetV2', color='lightcoral')
        bars2 = ax.bar(x + width/2, refined_mres, width, label='Refined', color='lightgreen')
        
        # Add improvement percentages
        for i, (name, stats) in enumerate(challenging_landmarks):
            improvement_pct = stats['improvement_percentage']
            ax.text(i, max(hrnet_mres[i], refined_mres[i]) + 0.2,
                   f'{improvement_pct:+.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Challenging Landmarks')
        ax.set_ylabel('MRE (pixels)')
        ax.set_title('Focus on Challenging Landmarks (>3px error)')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'challenging_landmarks_focus.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_results(self, output_dir: str):
        """Save detailed results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Overall metrics
        overall_df = pd.DataFrame([self.overall_metrics])
        overall_df.to_csv(os.path.join(output_dir, 'overall_results.csv'), index=False)
        
        # Per-landmark metrics
        per_landmark_df = pd.DataFrame([
            {
                'landmark_index': i,
                'landmark_name': name,
                'hrnet_mre': self.per_landmark_metrics[name]['hrnet_mre'],
                'hrnet_std': self.per_landmark_metrics[name]['hrnet_std'],
                'refined_mre': self.per_landmark_metrics[name]['refined_mre'],
                'refined_std': self.per_landmark_metrics[name]['refined_std'],
                'improvement_mean': self.per_landmark_metrics[name]['improvement_mean'],
                'improvement_percentage': self.per_landmark_metrics[name]['improvement_percentage'],
                'count': self.per_landmark_metrics[name]['count']
            }
            for i, name in enumerate(self.landmark_names) if name in self.per_landmark_metrics
        ])
        per_landmark_df.to_csv(os.path.join(output_dir, 'per_landmark_results.csv'), index=False)
        
        print(f"Results saved to {output_dir}")


def main():
    """Main evaluation function."""
    print("="*80)
    print("MLP REFINEMENT NETWORK EVALUATION")
    print("🔍 Comparing refined predictions with HRNetV2 baseline")
    print("="*80)
    
    # Initialize MMPose scope
    init_default_scope('mmpose')
    
    # Import custom modules
    import custom_cephalometric_dataset
    import custom_transforms
    
    # Configuration
    config = {
        'data_file': "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json",
        'hrnet_config': "Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py",
        'hrnet_checkpoint_pattern': "work_dirs/hrnetv2_w18_cephalometric_384x384_adaptive_wing_loss_v4/best_NME_epoch_*.pth",
        'mlp_checkpoint_pattern': "work_dirs/mlp_refinement_v1/best_model.pth",
        'input_size': 384,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
    }
    
    # Find checkpoints
    hrnet_checkpoints = glob.glob(config['hrnet_checkpoint_pattern'])
    mlp_checkpoints = glob.glob(config['mlp_checkpoint_pattern'])
    
    if not hrnet_checkpoints:
        print("❌ No HRNetV2 checkpoint found!")
        return
    
    if not mlp_checkpoints:
        print("❌ No MLP refinement checkpoint found!")
        return
    
    hrnet_checkpoint = max(hrnet_checkpoints, key=os.path.getctime)
    mlp_checkpoint = mlp_checkpoints[0]
    
    print(f"🔗 HRNetV2 checkpoint: {hrnet_checkpoint}")
    print(f"🧠 MLP checkpoint: {mlp_checkpoint}")
    
    # Load data
    print("\n📊 Loading test data...")
    main_df = pd.read_json(config['data_file'])
    test_df = main_df[main_df['set'] == 'test'].reset_index(drop=True)
    
    if test_df.empty:
        print("Test set empty, using validation set")
        test_df = main_df[main_df['set'] == 'dev'].reset_index(drop=True)
    
    print(f"Evaluation samples: {len(test_df)}")
    
    # Create dataset
    print("\n🔄 Creating evaluation dataset...")
    dataset = MLPRefinementDataset(
        test_df, config['hrnet_config'], hrnet_checkpoint,
        input_size=config['input_size'], cache_predictions=True
    )
    
    # Load MLP model
    print("\n🧠 Loading MLP refinement model...")
    model_config = {
        'num_landmarks': 19,
        'input_size': config['input_size'],
        'image_feature_dim': 512,
        'landmark_hidden_dims': [256, 128, 64],
        'dropout': 0.3,
        'use_landmark_weights': True
    }
    
    model = create_model(model_config)
    
    # Load checkpoint
    checkpoint = torch.load(mlp_checkpoint, map_location=config['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model loaded from epoch {checkpoint['epoch']}")
    
    # Create evaluator
    evaluator = MLPRefinementEvaluator(model, dataset, config['device'])
    
    # Run evaluation
    print("\n🎯 Running evaluation...")
    overall_metrics = evaluator.evaluate_all()
    
    # Print results
    evaluator.print_results()
    
    # Create output directory
    output_dir = "work_dirs/mlp_refinement_v1/evaluation_results"
    
    # Save results and plots
    evaluator.save_results(output_dir)
    evaluator.plot_results(output_dir)
    
    print(f"\n✅ Evaluation completed!")
    print(f"📈 Results summary:")
    print(f"  HRNetV2 MRE: {overall_metrics['hrnet_mre']:.3f} pixels")
    print(f"  Refined MRE: {overall_metrics['refined_mre']:.3f} pixels")
    print(f"  Improvement: {overall_metrics['improvement_mean']:+.3f} pixels ({overall_metrics['improvement_percentage']:+.1f}%)")
    print(f"\n📁 Detailed results saved to: {output_dir}")


if __name__ == "__main__":
    main() 