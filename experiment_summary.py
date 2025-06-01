#!/usr/bin/env python3
"""
Summary of all cephalometric landmark detection experiments.
Shows configuration and expected improvements for each experiment.
"""

import os
import pandas as pd

# Baseline results from V4 (AdaptiveWingLoss at 384x384)
BASELINE = {
    'name': 'V4 Baseline (AdaptiveWingLoss)',
    'resolution': '384x384',
    'loss': 'AdaptiveWingLoss',
    'overall_mre': 2.348,
    'sella_mre': 4.674,
    'gonion_mre': 4.281,
    'training_time': '~3 hours'
}

# Experiment configurations
EXPERIMENTS = {
    'A': {
        'name': 'AdaptiveWing + OHKM Hybrid',
        'resolution': '384x384',
        'loss': 'AdaptiveWingOHKMHybridLoss',
        'key_params': {
            'topk': 8,
            'ohkm_weight': 2.0,
            'batch_size': 20
        },
        'hypothesis': 'OHKM will focus training on hard landmarks (Sella/Gonion)',
        'expected_improvement': '2-5% overall, 10%+ on hard landmarks',
        'target_mre': '<2.3 pixels',
        'status': 'Ready to run',
        'script': 'train_experiment_a.py'
    },
    'B': {
        'name': 'FocalHeatmapLoss',
        'resolution': '384x384',
        'loss': 'FocalHeatmapLoss',
        'key_params': {
            'alpha': 0.25,
            'gamma': 2.0,
            'batch_size': 20
        },
        'hypothesis': 'Focal loss will improve by focusing on hard-to-predict pixels',
        'expected_improvement': '3-7% overall improvement',
        'target_mre': '<2.25 pixels',
        'status': 'Config pending',
        'script': 'train_experiment_b.py'
    },
    'C': {
        'name': 'OHKMMSELoss',
        'resolution': '384x384',
        'loss': 'OHKMMSELoss',
        'key_params': {
            'topk': 8,
            'sigma': 4,
            'batch_size': 20
        },
        'hypothesis': 'Pure OHKM with larger sigma for smoother gradients',
        'expected_improvement': '2-4% overall improvement',
        'target_mre': '<2.3 pixels',
        'status': 'Config pending',
        'script': 'train_experiment_c.py'
    },
    'D': {
        'name': 'CombinedTargetMSE 512x512',
        'resolution': '512x512',
        'loss': 'CombinedTargetMSELoss',
        'key_params': {
            'heatmap_weight': 1.0,
            'coord_weight': 0.5,
            'batch_size': 12  # Reduced for 512x512
        },
        'hypothesis': 'Higher resolution + coordinate regression for sub-pixel accuracy',
        'expected_improvement': '5-10% from resolution, 2-3% from coord regression',
        'target_mre': '<2.15 pixels',
        'status': 'Config pending',
        'script': 'train_experiment_d.py'
    }
}

def print_experiment_summary():
    """Print a formatted summary of all experiments."""
    
    print("="*100)
    print("CEPHALOMETRIC LANDMARK DETECTION - EXPERIMENT SUMMARY")
    print("="*100)
    
    # Baseline info
    print("\n📊 BASELINE RESULTS (V4)")
    print("-"*50)
    print(f"Model: HRNetV2-W18 with {BASELINE['loss']}")
    print(f"Resolution: {BASELINE['resolution']}")
    print(f"Overall MRE: {BASELINE['overall_mre']:.3f} pixels")
    print(f"Sella MRE: {BASELINE['sella_mre']:.3f} pixels")
    print(f"Gonion MRE: {BASELINE['gonion_mre']:.3f} pixels")
    print(f"Training Time: {BASELINE['training_time']}")
    
    # Experiments
    print("\n" + "="*100)
    print("🧪 PLANNED EXPERIMENTS")
    print("="*100)
    
    for exp_id, exp in EXPERIMENTS.items():
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {exp_id}: {exp['name']}")
        print(f"{'='*80}")
        print(f"Resolution: {exp['resolution']}")
        print(f"Loss Function: {exp['loss']}")
        print(f"Status: {exp['status']}")
        print(f"Training Script: {exp['script']}")
        
        print(f"\nKey Parameters:")
        for param, value in exp['key_params'].items():
            print(f"  • {param}: {value}")
        
        print(f"\nHypothesis: {exp['hypothesis']}")
        print(f"Expected Improvement: {exp['expected_improvement']}")
        print(f"Target MRE: {exp['target_mre']}")
    
    # Comparison table
    print("\n" + "="*100)
    print("📈 QUICK COMPARISON")
    print("="*100)
    
    data = []
    data.append({
        'Experiment': 'Baseline',
        'Resolution': BASELINE['resolution'],
        'Loss Type': 'AdaptiveWing',
        'Target MRE': f"{BASELINE['overall_mre']:.3f}",
        'Key Feature': 'Current best',
        'Status': 'Complete'
    })
    
    for exp_id, exp in EXPERIMENTS.items():
        data.append({
            'Experiment': f"Exp {exp_id}",
            'Resolution': exp['resolution'],
            'Loss Type': exp['loss'].replace('Loss', ''),
            'Target MRE': exp['target_mre'],
            'Key Feature': exp['name'].split()[0],
            'Status': exp['status']
    })
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    # Running instructions
    print("\n" + "="*100)
    print("🚀 HOW TO RUN EXPERIMENTS")
    print("="*100)
    print("\n1. For Experiment A (Ready to run):")
    print("   python train_experiment_a.py")
    print("\n2. To evaluate after training:")
    print("   python evaluate_experiment.py --experiment A")
    print("\n3. To create configs for B, C, D:")
    print("   python create_experiment_configs.py")
    
    # Time estimates
    print("\n" + "="*100)
    print("⏱️  TIME ESTIMATES")
    print("="*100)
    print("• Experiments A, B, C (384x384): ~3 hours each")
    print("• Experiment D (512x512): ~5-6 hours")
    print("• Total time for all experiments: ~15 hours")
    print("• Can run in parallel on multiple GPUs")

def check_experiment_status():
    """Check which experiments have been configured and trained."""
    status = {}
    
    for exp_id in EXPERIMENTS:
        exp_status = {
            'config_exists': False,
            'script_exists': False,
            'checkpoint_exists': False,
            'results_exist': False
        }
        
        # Check config
        config_path = f"configs/experiment_{exp_id.lower()}_*.py"
        import glob
        if glob.glob(config_path):
            exp_status['config_exists'] = True
        
        # Check training script
        script_path = f"train_experiment_{exp_id.lower()}.py"
        if os.path.exists(script_path):
            exp_status['script_exists'] = True
        
        # Check for checkpoints
        work_dir = EXPERIMENTS[exp_id].get('work_dir', f'work_dirs/experiment_{exp_id.lower()}_*')
        checkpoint_pattern = f"{work_dir}/best_NME_epoch_*.pth"
        if glob.glob(checkpoint_pattern):
            exp_status['checkpoint_exists'] = True
        
        # Check for results
        results_pattern = f"{work_dir}/evaluation_results/experiment_{exp_id}_summary.csv"
        if glob.glob(results_pattern):
            exp_status['results_exist'] = True
        
        status[exp_id] = exp_status
    
    return status

def main():
    """Main function to display experiment summary."""
    print_experiment_summary()
    
    # Check actual status
    print("\n" + "="*100)
    print("📋 CURRENT STATUS CHECK")
    print("="*100)
    
    status = check_experiment_status()
    
    for exp_id, exp_status in status.items():
        print(f"\nExperiment {exp_id}:")
        print(f"  ✓ Config: {'Yes' if exp_status['config_exists'] else 'No'}")
        print(f"  ✓ Script: {'Yes' if exp_status['script_exists'] else 'No'}")
        print(f"  ✓ Trained: {'Yes' if exp_status['checkpoint_exists'] else 'No'}")
        print(f"  ✓ Evaluated: {'Yes' if exp_status['results_exist'] else 'No'}")

if __name__ == "__main__":
    main() 