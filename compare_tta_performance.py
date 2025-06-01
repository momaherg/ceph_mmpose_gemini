#!/usr/bin/env python3
"""
Comparison script to evaluate performance improvement with TTA.
Runs both regular and TTA evaluation and compares results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import sys
import argparse
from pathlib import Path

def run_evaluation(script_path, checkpoint_path, output_dir, use_tta=False):
    """Run evaluation script and return results."""
    cmd = [
        sys.executable, script_path,
        "--checkpoint", checkpoint_path,
        "--output_dir", output_dir
    ]
    
    print(f"Running {'TTA' if use_tta else 'regular'} evaluation...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        if result.returncode != 0:
            print(f"Error running evaluation: {result.stderr}")
            return None
        return True
    except subprocess.TimeoutExpired:
        print("Evaluation timed out")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def load_results(results_dir, is_tta=False):
    """Load evaluation results from CSV files."""
    suffix = "_tta" if is_tta else ""
    
    # Load per-landmark results
    landmark_file = f"per_landmark_mre_results{suffix}.csv"
    landmark_path = os.path.join(results_dir, landmark_file)
    
    # Load overall summary
    summary_file = f"overall_summary{suffix}.csv"
    summary_path = os.path.join(results_dir, summary_file)
    
    if not os.path.exists(landmark_path) or not os.path.exists(summary_path):
        print(f"Results files not found in {results_dir}")
        return None, None
        
    landmark_df = pd.read_csv(landmark_path)
    summary_df = pd.read_csv(summary_path)
    
    return landmark_df, summary_df

def create_comparison_plots(regular_landmarks, tta_landmarks, regular_summary, tta_summary, output_dir):
    """Create comparison plots showing TTA improvements."""
    
    # Extract MRE values
    if 'mre_pixels' in regular_landmarks.columns:
        regular_mre = regular_landmarks['mre_pixels'].values
    else:
        regular_mre = regular_landmarks.iloc[:, 2].values  # Assume 3rd column is MRE
        
    if 'mre_pixels_tta' in tta_landmarks.columns:
        tta_mre = tta_landmarks['mre_pixels_tta'].values
    else:
        tta_mre = tta_landmarks.iloc[:, 2].values  # Assume 3rd column is MRE
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Per-landmark comparison
    landmark_indices = range(len(regular_mre))
    width = 0.35
    x = np.arange(len(landmark_indices))
    
    ax1.bar(x - width/2, regular_mre, width, label='Regular', alpha=0.7, color='lightcoral')
    ax1.bar(x + width/2, tta_mre, width, label='TTA', alpha=0.7, color='lightgreen')
    
    ax1.set_xlabel('Landmark Index')
    ax1.set_ylabel('MRE (pixels)')
    ax1.set_title('Per-Landmark MRE Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(i) for i in landmark_indices], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Improvement percentage per landmark
    improvement = ((regular_mre - tta_mre) / regular_mre) * 100
    colors = ['green' if imp > 0 else 'red' for imp in improvement]
    
    ax2.bar(landmark_indices, improvement, color=colors, alpha=0.7)
    ax2.set_xlabel('Landmark Index')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('TTA Improvement per Landmark')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Add text annotations for significant improvements
    for i, imp in enumerate(improvement):
        if abs(imp) > 5:  # Show label if improvement > 5%
            ax2.text(i, imp + (1 if imp > 0 else -1), f'{imp:.1f}%', 
                    ha='center', va='bottom' if imp > 0 else 'top', fontsize=8)
    
    # 3. Overall MRE comparison
    regular_overall = regular_summary.iloc[0, 0] if len(regular_summary) > 0 else 0
    tta_overall = tta_summary.iloc[0, 0] if len(tta_summary) > 0 else 0
    
    overall_methods = ['Regular', 'TTA']
    overall_mres = [regular_overall, tta_overall]
    
    bars = ax3.bar(overall_methods, overall_mres, color=['lightcoral', 'lightgreen'], alpha=0.7)
    ax3.set_ylabel('Overall MRE (pixels)')
    ax3.set_title('Overall MRE Comparison')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, overall_mres):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 4. Improvement distribution
    ax4.hist(improvement, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_xlabel('Improvement (%)')
    ax4.set_ylabel('Number of Landmarks')
    ax4.set_title('Distribution of TTA Improvements')
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No improvement')
    ax4.axvline(x=np.mean(improvement), color='green', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(improvement):.1f}%')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "tta_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_path}")
    plt.close()
    
    return improvement

def create_summary_report(regular_landmarks, tta_landmarks, regular_summary, tta_summary, improvement, output_dir):
    """Create a detailed summary report."""
    
    # Calculate statistics
    regular_overall = regular_summary.iloc[0, 0] if len(regular_summary) > 0 else 0
    tta_overall = tta_summary.iloc[0, 0] if len(tta_summary) > 0 else 0
    overall_improvement = ((regular_overall - tta_overall) / regular_overall) * 100
    
    # Statistics about per-landmark improvements
    positive_improvements = improvement[improvement > 0]
    negative_improvements = improvement[improvement < 0]
    
    landmarks_improved = len(positive_improvements)
    landmarks_degraded = len(negative_improvements)
    landmarks_total = len(improvement)
    
    report = f"""
TTA EVALUATION COMPARISON REPORT
{'='*50}

OVERALL RESULTS:
- Regular MRE: {regular_overall:.3f} pixels
- TTA MRE: {tta_overall:.3f} pixels
- Overall improvement: {overall_improvement:.2f}%

PER-LANDMARK ANALYSIS:
- Total landmarks: {landmarks_total}
- Landmarks improved: {landmarks_improved} ({landmarks_improved/landmarks_total*100:.1f}%)
- Landmarks degraded: {landmarks_degraded} ({landmarks_degraded/landmarks_total*100:.1f}%)

IMPROVEMENT STATISTICS:
- Mean improvement: {np.mean(improvement):.2f}%
- Median improvement: {np.median(improvement):.2f}%
- Best improvement: {np.max(improvement):.2f}%
- Worst degradation: {np.min(improvement):.2f}%
- Standard deviation: {np.std(improvement):.2f}%

TOP 5 MOST IMPROVED LANDMARKS:
"""
    
    # Add top improvements
    sorted_indices = np.argsort(improvement)[::-1]  # Sort descending
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[i]
        landmark_name = regular_landmarks.iloc[idx, 1] if len(regular_landmarks.columns) > 1 else f"Landmark_{idx}"
        report += f"- {landmark_name}: {improvement[idx]:.2f}% improvement\n"
    
    # Add bottom improvements (worst degradations)
    if np.any(improvement < 0):
        report += "\nTOP 5 MOST DEGRADED LANDMARKS:\n"
        for i in range(max(0, len(sorted_indices)-5), len(sorted_indices)):
            idx = sorted_indices[i]
            landmark_name = regular_landmarks.iloc[idx, 1] if len(regular_landmarks.columns) > 1 else f"Landmark_{idx}"
            report += f"- {landmark_name}: {improvement[idx]:.2f}% change\n"
    
    # Save report
    report_path = os.path.join(output_dir, "tta_comparison_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"Detailed report saved to: {report_path}")

def main():
    """Main comparison function."""
    
    parser = argparse.ArgumentParser(description='Compare regular vs TTA evaluation performance')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--output_dir', type=str, help='Output directory for comparison results')
    parser.add_argument('--skip_regular', action='store_true', help='Skip regular evaluation (use existing results)')
    parser.add_argument('--skip_tta', action='store_true', help='Skip TTA evaluation (use existing results)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint file not found: {args.checkpoint}")
        return
    
    # Set up output directories
    base_output_dir = args.output_dir or "comparison_results"
    os.makedirs(base_output_dir, exist_ok=True)
    
    regular_output_dir = os.path.join(base_output_dir, "regular_evaluation")
    tta_output_dir = os.path.join(base_output_dir, "tta_evaluation")
    
    os.makedirs(regular_output_dir, exist_ok=True)
    os.makedirs(tta_output_dir, exist_ok=True)
    
    print("="*80)
    print("TTA PERFORMANCE COMPARISON")
    print("="*80)
    
    # Run evaluations
    if not args.skip_regular:
        print("\n1. Running regular evaluation...")
        success = run_evaluation("evaluate_detailed_metrics.py", args.checkpoint, regular_output_dir, use_tta=False)
        if not success:
            print("Regular evaluation failed")
            return
    else:
        print("\n1. Skipping regular evaluation (using existing results)")
    
    if not args.skip_tta:
        print("\n2. Running TTA evaluation...")
        success = run_evaluation("evaluate_with_tta.py", args.checkpoint, tta_output_dir, use_tta=True)
        if not success:
            print("TTA evaluation failed")
            return
    else:
        print("\n2. Skipping TTA evaluation (using existing results)")
    
    # Load results
    print("\n3. Loading and comparing results...")
    
    regular_landmarks, regular_summary = load_results(regular_output_dir, is_tta=False)
    tta_landmarks, tta_summary = load_results(tta_output_dir, is_tta=True)
    
    if regular_landmarks is None or tta_landmarks is None:
        print("Failed to load evaluation results")
        return
    
    # Create comparison visualizations
    print("\n4. Creating comparison plots...")
    improvement = create_comparison_plots(regular_landmarks, tta_landmarks, 
                                        regular_summary, tta_summary, base_output_dir)
    
    # Create summary report
    print("\n5. Generating summary report...")
    create_summary_report(regular_landmarks, tta_landmarks, regular_summary, 
                         tta_summary, improvement, base_output_dir)
    
    print(f"\nComparison complete! Results saved to: {base_output_dir}")

if __name__ == "__main__":
    main() 