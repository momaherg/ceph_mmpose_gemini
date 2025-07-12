#!/usr/bin/env python3
"""
Test script for evaluating HRNetV2 with classification head.
This script shows how to use the evaluation script.
"""

import os
import subprocess
import sys

def main():
    """Run evaluation with different configurations."""
    
    print("="*80)
    print("TEST: HRNetV2 WITH CLASSIFICATION EVALUATION")
    print("="*80)
    
    # Check if we have a trained model
    work_dir = "work_dirs/hrnetv2_w18_cephalometric_ensemble_concurrent_mlp_v5"
    
    if not os.path.exists(work_dir):
        print(f"ERROR: Work directory not found: {work_dir}")
        print("Please train a model first using train_concurrent_v5.py")
        return
    
    # Check for model directories
    model_dirs = []
    for i in range(1, 4):  # Check for 3 models
        model_dir = os.path.join(work_dir, f"model_{i}")
        if os.path.exists(model_dir):
            model_dirs.append((i, model_dir))
    
    if not model_dirs:
        print("ERROR: No trained models found")
        print("Please train models first")
        return
    
    print(f"Found {len(model_dirs)} trained models")
    
    # Evaluate each model
    for model_idx, model_dir in model_dirs:
        print(f"\n{'='*60}")
        print(f"Evaluating Model {model_idx}")
        print(f"{'='*60}")
        
        # Check for checkpoints
        import glob
        checkpoints = glob.glob(os.path.join(model_dir, "epoch_*.pth"))
        if not checkpoints:
            print(f"No checkpoints found in {model_dir}")
            continue
        
        # Get latest checkpoint
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        epoch = int(latest_checkpoint.split('_')[-1].split('.')[0])
        
        print(f"Using checkpoint: epoch_{epoch}.pth")
        
        # Run evaluation
        cmd = [
            sys.executable,
            "evaluate_classification_model.py",
            "--work_dir", work_dir,
            "--model_idx", str(model_idx),
            "--epoch", str(epoch)
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Print key results
            output_lines = result.stdout.split('\n')
            
            # Extract key metrics
            for line in output_lines:
                if "MRE (pixels)" in line or "Overall Accuracy" in line:
                    print(f"  {line.strip()}")
                elif "Native vs Post-hoc Comparison:" in line:
                    print(f"\n{line.strip()}")
                elif "Accuracy Difference:" in line or "Agreement Rate:" in line:
                    print(f"  {line.strip()}")
            
            # Check if evaluation results were saved
            eval_dir = os.path.join(model_dir, "classification_evaluation")
            if os.path.exists(eval_dir):
                print(f"\n✓ Results saved in: {eval_dir}")
                
                # List saved files
                saved_files = os.listdir(eval_dir)
                print("  Saved files:")
                for file in saved_files:
                    print(f"    - {file}")
                    
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Evaluation failed for model {model_idx}")
            print(f"Error output: {e.stderr}")
            continue
    
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    # Summarize results across models
    print("\nTo view detailed results for each model:")
    for model_idx, model_dir in model_dirs:
        eval_dir = os.path.join(model_dir, "classification_evaluation")
        if os.path.exists(eval_dir):
            print(f"  Model {model_idx}: {eval_dir}/evaluation_results.json")
    
    print("\nTo view confusion matrices:")
    for model_idx, model_dir in model_dirs:
        eval_dir = os.path.join(model_dir, "classification_evaluation")
        if os.path.exists(eval_dir):
            print(f"  Model {model_idx}:")
            print(f"    - Native: {eval_dir}/confusion_matrix_native.png")
            print(f"    - Post-hoc: {eval_dir}/confusion_matrix_posthoc.png")
            print(f"    - Comparison: {eval_dir}/classification_comparison.png")
    
    print("\n💡 USAGE TIPS:")
    print("1. To evaluate a specific epoch:")
    print("   python evaluate_classification_model.py --epoch 20")
    print("\n2. To evaluate with a custom test split:")
    print("   python evaluate_classification_model.py --test_split_file data/test_patients.txt")
    print("\n3. To evaluate a single model from the ensemble:")
    print("   python evaluate_classification_model.py --model_idx 1")


if __name__ == "__main__":
    main() 