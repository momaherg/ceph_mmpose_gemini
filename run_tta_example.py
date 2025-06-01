#!/usr/bin/env python3
"""
Simple example script to run TTA evaluation.
This demonstrates the easiest way to get started with TTA evaluation.
"""

import os
import glob
import subprocess
import sys

def find_best_checkpoint(work_dir="work_dirs/hrnetv2_w18_cephalometric_384x384_adaptive_wing_loss_v4"):
    """Find the best checkpoint in the work directory."""
    
    # Look for best checkpoint first
    pattern = os.path.join(work_dir, "best_NME_epoch_*.pth")
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        # Fallback to any epoch checkpoint
        pattern = os.path.join(work_dir, "epoch_*.pth")
        checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None
    
    # Return the most recent checkpoint
    return max(checkpoints, key=os.path.getctime)

def run_tta_evaluation():
    """Run TTA evaluation with automatic checkpoint detection."""
    
    print("="*60)
    print("SIMPLE TTA EVALUATION EXAMPLE")
    print("="*60)
    
    # Find checkpoint
    checkpoint = find_best_checkpoint()
    if not checkpoint:
        print("❌ No checkpoint found in default work directory!")
        print("   Make sure you have trained a model first.")
        print("   Or specify a checkpoint manually using:")
        print("   python evaluate_with_tta.py --checkpoint /path/to/checkpoint.pth")
        return False
    
    print(f"✅ Found checkpoint: {checkpoint}")
    
    # Run TTA evaluation
    print("\n🚀 Running TTA evaluation...")
    print("   This will apply ~20 different augmentations and average the results")
    print("   Expected runtime: 2-5 minutes for ~60 test images")
    
    try:
        cmd = [
            sys.executable, "evaluate_with_tta.py",
            "--checkpoint", checkpoint,
            "--output_dir", "tta_example_results"
        ]
        
        print(f"\n📝 Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        
        print("\n✅ TTA evaluation completed successfully!")
        print("📊 Results saved to: tta_example_results/")
        print("\n📈 Key output files:")
        print("   - per_landmark_mre_results_tta.csv (detailed per-landmark metrics)")
        print("   - overall_summary_tta.csv (overall statistics)")
        print("   - error_analysis_tta.png (visualization plots)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ TTA evaluation failed with error: {e}")
        return False
    except FileNotFoundError:
        print("❌ evaluate_with_tta.py not found in current directory!")
        return False

def run_comparison():
    """Run comparison between regular and TTA evaluation."""
    
    checkpoint = find_best_checkpoint()
    if not checkpoint:
        print("❌ No checkpoint found for comparison!")
        return False
    
    print("\n🔄 Running comparison between regular and TTA evaluation...")
    print("   This will run both evaluations and show improvement metrics")
    
    try:
        cmd = [
            sys.executable, "compare_tta_performance.py",
            "--checkpoint", checkpoint,
            "--output_dir", "tta_comparison_example"
        ]
        
        print(f"\n📝 Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        
        print("\n✅ Comparison completed successfully!")
        print("📊 Results saved to: tta_comparison_example/")
        print("\n📈 Key output files:")
        print("   - tta_comparison.png (visual comparison)")
        print("   - tta_comparison_report.txt (detailed improvement analysis)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Comparison failed with error: {e}")
        return False

def main():
    """Main example function."""
    
    print("TTA Evaluation Example")
    print("Choose an option:")
    print("1. Run TTA evaluation only")
    print("2. Run comparison (regular vs TTA)")
    print("3. Both")
    
    try:
        choice = input("\nEnter choice (1/2/3): ").strip()
    except KeyboardInterrupt:
        print("\nExiting...")
        return
    
    if choice == "1":
        run_tta_evaluation()
    elif choice == "2":
        run_comparison()
    elif choice == "3":
        if run_tta_evaluation():
            print("\n" + "="*60)
            run_comparison()
    else:
        print("Invalid choice. Running TTA evaluation by default...")
        run_tta_evaluation()

if __name__ == "__main__":
    main() 