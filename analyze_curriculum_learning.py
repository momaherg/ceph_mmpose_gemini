#!/usr/bin/env python3
"""
Curriculum Learning & Hard-Example Analysis Script
This script analyzes the effectiveness of curriculum augmentation and hard-example
oversampling during concurrent MLP training.
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import argparse

def load_hard_example_stats(work_dir: str) -> List[Dict[str, Any]]:
    """Load hard-example statistics from all epochs."""
    stats_dir = os.path.join(work_dir, "concurrent_mlp")
    
    if not os.path.exists(stats_dir):
        print(f"❌ Stats directory not found: {stats_dir}")
        return []
    
    # Find all hard-example stats files
    stats_files = glob.glob(os.path.join(stats_dir, "hard_example_stats_epoch_*.json"))
    
    if not stats_files:
        print(f"❌ No hard-example stats files found in {stats_dir}")
        return []
    
    all_stats = []
    
    for stats_file in sorted(stats_files):
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                all_stats.append(stats)
        except Exception as e:
            print(f"⚠️  Failed to load {stats_file}: {e}")
    
    return all_stats

def analyze_hard_example_evolution(stats_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Analyze how hard examples evolve over training."""
    data = []
    
    for stats in stats_list:
        data.append({
            'epoch': stats['epoch'],
            'total_samples': stats['total_samples'],
            'hard_examples': stats['hard_examples'],
            'hard_example_ratio': stats['hard_example_ratio'],
            'avg_sample_error': stats['avg_sample_error'],
            'curriculum_active': stats['curriculum_active']
        })
    
    return pd.DataFrame(data)

def plot_curriculum_effectiveness(df: pd.DataFrame, output_dir: str):
    """Create comprehensive plots showing curriculum learning effectiveness."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Curriculum Learning & Hard-Example Analysis', fontsize=16, fontweight='bold')
    
    # 1. Hard examples over time
    ax = axes[0, 0]
    ax.plot(df['epoch'], df['hard_examples'], 'o-', color='red', linewidth=2, markersize=6)
    ax.set_xlabel('HRNet Epoch')
    ax.set_ylabel('Number of Hard Examples')
    ax.set_title('Hard Examples Over Training')
    ax.grid(True, alpha=0.3)
    
    # Add curriculum start line
    curriculum_start = df[df['curriculum_active']]['epoch'].min() if any(df['curriculum_active']) else None
    if curriculum_start:
        ax.axvline(x=curriculum_start, color='green', linestyle='--', alpha=0.7, 
                  label=f'Curriculum starts (epoch {curriculum_start})')
        ax.legend()
    
    # 2. Hard example ratio over time
    ax = axes[0, 1]
    ax.plot(df['epoch'], df['hard_example_ratio'] * 100, 'o-', color='orange', linewidth=2, markersize=6)
    ax.set_xlabel('HRNet Epoch')
    ax.set_ylabel('Hard Example Ratio (%)')
    ax.set_title('Hard Example Ratio Over Training')
    ax.grid(True, alpha=0.3)
    
    if curriculum_start:
        ax.axvline(x=curriculum_start, color='green', linestyle='--', alpha=0.7)
    
    # 3. Average sample error over time
    ax = axes[0, 2]
    ax.plot(df['epoch'], df['avg_sample_error'], 'o-', color='blue', linewidth=2, markersize=6)
    ax.set_xlabel('HRNet Epoch')
    ax.set_ylabel('Average Sample MRE (pixels)')
    ax.set_title('Average Sample Error Over Training')
    ax.grid(True, alpha=0.3)
    
    if curriculum_start:
        ax.axvline(x=curriculum_start, color='green', linestyle='--', alpha=0.7)
    
    # 4. Curriculum impact analysis
    ax = axes[1, 0]
    if curriculum_start:
        pre_curriculum = df[df['epoch'] < curriculum_start]
        post_curriculum = df[df['epoch'] >= curriculum_start]
        
        if not pre_curriculum.empty and not post_curriculum.empty:
            pre_avg_error = pre_curriculum['avg_sample_error'].mean()
            post_avg_error = post_curriculum['avg_sample_error'].mean()
            
            improvement = (pre_avg_error - post_avg_error) / pre_avg_error * 100
            
            ax.bar(['Pre-Curriculum', 'Post-Curriculum'], 
                  [pre_avg_error, post_avg_error],
                  color=['lightcoral', 'lightgreen'],
                  alpha=0.7)
            ax.set_ylabel('Average Sample MRE (pixels)')
            ax.set_title(f'Curriculum Impact\n({improvement:.1f}% improvement)')
            ax.grid(True, alpha=0.3)
            
            # Add improvement text
            ax.text(0.5, max(pre_avg_error, post_avg_error) * 0.9,
                   f'{improvement:.1f}% improvement',
                   ha='center', va='center', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    else:
        ax.text(0.5, 0.5, 'Curriculum not started yet', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Curriculum Impact (Not Available)')
    
    # 5. Hard example reduction rate
    ax = axes[1, 1]
    if len(df) > 1:
        # Calculate rolling improvement in hard example ratio
        window = min(3, len(df))
        rolling_hard_ratio = df['hard_example_ratio'].rolling(window=window).mean()
        improvement_rate = -rolling_hard_ratio.diff()  # Negative diff means reduction
        
        ax.plot(df['epoch'][window-1:], improvement_rate[window-1:], 'o-', 
               color='purple', linewidth=2, markersize=6)
        ax.set_xlabel('HRNet Epoch')
        ax.set_ylabel('Hard Example Reduction Rate')
        ax.set_title(f'Hard Example Reduction Rate\n(Rolling {window}-epoch window)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        if curriculum_start:
            ax.axvline(x=curriculum_start, color='green', linestyle='--', alpha=0.7)
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Hard Example Reduction Rate')
    
    # 6. Training efficiency metrics
    ax = axes[1, 2]
    if len(df) > 1:
        # Calculate effective training samples (with oversampling)
        effective_samples = df['total_samples'] + df['hard_examples']  # Simplified calculation
        efficiency = df['total_samples'] / effective_samples * 100
        
        ax.plot(df['epoch'], efficiency, 'o-', color='teal', linewidth=2, markersize=6)
        ax.set_xlabel('HRNet Epoch')
        ax.set_ylabel('Training Efficiency (%)')
        ax.set_title('Training Efficiency\n(Original/Effective Samples)')
        ax.grid(True, alpha=0.3)
        
        if curriculum_start:
            ax.axvline(x=curriculum_start, color='green', linestyle='--', alpha=0.7)
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Training Efficiency')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "curriculum_learning_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path

def generate_curriculum_report(df: pd.DataFrame, stats_list: List[Dict[str, Any]], output_dir: str):
    """Generate a comprehensive curriculum learning report."""
    
    report_path = os.path.join(output_dir, "curriculum_learning_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Curriculum Learning & Hard-Example Analysis Report\n\n")
        
        # Basic statistics
        f.write("## Training Overview\n\n")
        f.write(f"- **Total Epochs Analyzed**: {len(df)}\n")
        f.write(f"- **Total Training Samples**: {df['total_samples'].iloc[0] if not df.empty else 'N/A'}\n")
        f.write(f"- **Hard Example Threshold**: {stats_list[0].get('hard_example_threshold', 'N/A')} pixels\n\n")
        
        # Curriculum effectiveness
        curriculum_epochs = df[df['curriculum_active']]
        if not curriculum_epochs.empty:
            curriculum_start = curriculum_epochs['epoch'].min()
            f.write("## Curriculum Learning Effectiveness\n\n")
            f.write(f"- **Curriculum Start Epoch**: {curriculum_start}\n")
            
            pre_curriculum = df[df['epoch'] < curriculum_start]
            post_curriculum = df[df['epoch'] >= curriculum_start]
            
            if not pre_curriculum.empty and not post_curriculum.empty:
                pre_avg = pre_curriculum['avg_sample_error'].mean()
                post_avg = post_curriculum['avg_sample_error'].mean()
                improvement = (pre_avg - post_avg) / pre_avg * 100
                
                f.write(f"- **Pre-Curriculum Average Error**: {pre_avg:.3f} pixels\n")
                f.write(f"- **Post-Curriculum Average Error**: {post_avg:.3f} pixels\n")
                f.write(f"- **Overall Improvement**: {improvement:.1f}%\n\n")
            
            # Curriculum progression
            f.write("### Curriculum Progression\n\n")
            f.write("| Epoch | Hard Examples | Hard Ratio (%) | Avg Error (px) | Curriculum Active |\n")
            f.write("|-------|---------------|----------------|----------------|-------------------|\n")
            
            for _, row in df.iterrows():
                f.write(f"| {row['epoch']} | {row['hard_examples']} | "
                       f"{row['hard_example_ratio']*100:.1f} | {row['avg_sample_error']:.3f} | "
                       f"{'✓' if row['curriculum_active'] else '✗'} |\n")
        else:
            f.write("## Curriculum Learning Status\n\n")
            f.write("- **Status**: Curriculum learning has not started yet\n")
            f.write("- **Reason**: Training epochs are below curriculum start threshold\n\n")
        
        # Hard example analysis
        f.write("\n## Hard Example Analysis\n\n")
        
        if not df.empty:
            total_hard = df['hard_examples'].sum()
            avg_hard_ratio = df['hard_example_ratio'].mean() * 100
            max_hard_epoch = df.loc[df['hard_examples'].idxmax(), 'epoch']
            min_hard_epoch = df.loc[df['hard_examples'].idxmin(), 'epoch']
            
            f.write(f"- **Total Hard Examples Encountered**: {total_hard}\n")
            f.write(f"- **Average Hard Example Ratio**: {avg_hard_ratio:.1f}%\n")
            f.write(f"- **Epoch with Most Hard Examples**: {max_hard_epoch} ({df['hard_examples'].max()} examples)\n")
            f.write(f"- **Epoch with Fewest Hard Examples**: {min_hard_epoch} ({df['hard_examples'].min()} examples)\n\n")
            
            # Trend analysis
            if len(df) > 1:
                hard_trend = np.polyfit(df['epoch'], df['hard_examples'], 1)[0]
                error_trend = np.polyfit(df['epoch'], df['avg_sample_error'], 1)[0]
                
                f.write("### Trend Analysis\n\n")
                f.write(f"- **Hard Examples Trend**: {'Decreasing' if hard_trend < 0 else 'Increasing'} "
                       f"({hard_trend:.2f} examples/epoch)\n")
                f.write(f"- **Average Error Trend**: {'Decreasing' if error_trend < 0 else 'Increasing'} "
                       f"({error_trend:.4f} pixels/epoch)\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        if not curriculum_epochs.empty:
            recent_hard_ratio = df['hard_example_ratio'].iloc[-1] * 100
            
            if recent_hard_ratio > 20:
                f.write("- **High Hard Example Ratio**: Consider lowering the hard example threshold\n")
            elif recent_hard_ratio < 5:
                f.write("- **Low Hard Example Ratio**: Consider raising the hard example threshold for more aggressive training\n")
            
            if len(df) > 5:
                recent_improvement = (df['avg_sample_error'].iloc[-5:].iloc[0] - df['avg_sample_error'].iloc[-1]) / df['avg_sample_error'].iloc[-5:].iloc[0] * 100
                
                if recent_improvement < 1:
                    f.write("- **Slow Recent Improvement**: Consider adjusting curriculum parameters or augmentation intensity\n")
                elif recent_improvement > 10:
                    f.write("- **Excellent Recent Improvement**: Current curriculum settings are working well\n")
        else:
            f.write("- **Curriculum Not Started**: Wait for more epochs to see curriculum learning effects\n")
        
        f.write("\n---\n")
        f.write(f"*Report generated from {len(df)} training epochs*\n")
    
    return report_path

def main():
    """Main analysis function."""
    
    parser = argparse.ArgumentParser(description='Analyze Curriculum Learning & Hard-Example Oversampling')
    parser.add_argument(
        '--work_dir',
        type=str,
        default='work_dirs/hrnetv2_w18_cephalometric_concurrent_mlp_v5',
        help='Work directory containing concurrent MLP training results'
    )
    args = parser.parse_args()
    
    print("="*80)
    print("CURRICULUM LEARNING & HARD-EXAMPLE ANALYSIS")
    print("="*80)
    
    # Load hard-example statistics
    print(f"📊 Loading statistics from: {args.work_dir}")
    stats_list = load_hard_example_stats(args.work_dir)
    
    if not stats_list:
        print("❌ No statistics found. Make sure concurrent training has run for at least one epoch.")
        return
    
    print(f"✓ Loaded statistics from {len(stats_list)} epochs")
    
    # Analyze evolution
    df = analyze_hard_example_evolution(stats_list)
    print(f"✓ Analyzed evolution across {len(df)} epochs")
    
    # Create output directory
    output_dir = os.path.join(args.work_dir, "curriculum_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    print("📈 Generating analysis plots...")
    plot_path = plot_curriculum_effectiveness(df, output_dir)
    print(f"✓ Plots saved to: {plot_path}")
    
    # Generate report
    print("📝 Generating comprehensive report...")
    report_path = generate_curriculum_report(df, stats_list, output_dir)
    print(f"✓ Report saved to: {report_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    if not df.empty:
        latest_stats = df.iloc[-1]
        
        print(f"📊 **Latest Epoch**: {latest_stats['epoch']}")
        print(f"🎯 **Hard Examples**: {latest_stats['hard_examples']} ({latest_stats['hard_example_ratio']*100:.1f}%)")
        print(f"📉 **Average Error**: {latest_stats['avg_sample_error']:.3f} pixels")
        print(f"🎓 **Curriculum Active**: {'Yes' if latest_stats['curriculum_active'] else 'No'}")
        
        # Calculate overall improvement
        if len(df) > 1:
            initial_error = df['avg_sample_error'].iloc[0]
            latest_error = df['avg_sample_error'].iloc[-1]
            overall_improvement = (initial_error - latest_error) / initial_error * 100
            
            print(f"📈 **Overall Improvement**: {overall_improvement:.1f}%")
            
            # Trend analysis
            if len(df) >= 3:
                recent_trend = np.polyfit(df['epoch'].iloc[-3:], df['avg_sample_error'].iloc[-3:], 1)[0]
                trend_direction = "📉 Decreasing" if recent_trend < 0 else "📈 Increasing"
                print(f"🔄 **Recent Trend**: {trend_direction} ({recent_trend:.4f} pixels/epoch)")
        
        # Curriculum effectiveness
        curriculum_epochs = df[df['curriculum_active']]
        if not curriculum_epochs.empty:
            curriculum_start = curriculum_epochs['epoch'].min()
            pre_curriculum = df[df['epoch'] < curriculum_start]
            post_curriculum = df[df['epoch'] >= curriculum_start]
            
            if not pre_curriculum.empty and not post_curriculum.empty:
                pre_avg = pre_curriculum['avg_sample_error'].mean()
                post_avg = post_curriculum['avg_sample_error'].mean()
                curriculum_improvement = (pre_avg - post_avg) / pre_avg * 100
                
                print(f"🎓 **Curriculum Improvement**: {curriculum_improvement:.1f}%")
    
    print(f"\n💾 **Results saved to**: {output_dir}")
    print(f"   - Analysis plots: curriculum_learning_analysis.png")
    print(f"   - Detailed report: curriculum_learning_report.md")
    
    print("\n🎉 Analysis completed!")

if __name__ == "__main__":
    main() 