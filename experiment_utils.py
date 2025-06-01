"""
Utility functions for managing cephalometric landmark detection experiments.
"""

import os
import json
import glob
import pandas as pd
from experiments_config import experiments

def list_experiments():
    """List all available experiments with their status."""
    print("\n" + "="*100)
    print(f"{'ID':<3} {'Name':<30} {'Status':<15} {'Best NME':<12} {'Description'}")
    print("="*100)
    
    for i, exp in enumerate(experiments):
        work_dir = f"work_dirs/experiment_{i}_{exp['name']}"
        status = "Not Started"
        best_nme = "N/A"
        
        if os.path.exists(work_dir):
            # Check if training completed
            checkpoints = glob.glob(os.path.join(work_dir, "best_NME_epoch_*.pth"))
            if checkpoints:
                status = "Completed"
                # Try to get best NME from logs
                try:
                    log_file = os.path.join(work_dir, "vis_data", "scalars.json")
                    if os.path.exists(log_file):
                        with open(log_file, 'r') as f:
                            logs = [json.loads(line) for line in f]
                        nme_values = [log['NME'] for log in logs if 'NME' in log and log.get('mode') == 'val']
                        if nme_values:
                            best_nme = f"{min(nme_values):.4f}"
                except:
                    pass
            else:
                # Check if error occurred
                if os.path.exists(os.path.join(work_dir, "experiment_error.txt")):
                    status = "Failed"
                else:
                    status = "In Progress"
        
        print(f"{i:<3} {exp['name']:<30} {status:<15} {best_nme:<12} {exp['description']}")
    
    print("="*100)

def compare_experiments(indices=None):
    """Compare results across experiments."""
    if indices is None:
        # Compare all completed experiments
        indices = []
        for i in range(len(experiments)):
            work_dir = f"work_dirs/experiment_{i}_{experiments[i]['name']}"
            if os.path.exists(work_dir) and glob.glob(os.path.join(work_dir, "best_NME_epoch_*.pth")):
                indices.append(i)
    
    if not indices:
        print("No completed experiments to compare.")
        return
    
    results = []
    
    for idx in indices:
        exp = experiments[idx]
        work_dir = f"work_dirs/experiment_{idx}_{exp['name']}"
        
        result = {
            'ID': idx,
            'Name': exp['name'],
            'Input Size': str(exp['config']['input_size']),
            'Loss': exp['config']['loss_type'],
            'Optimizer': exp['config']['optimizer'],
            'LR': exp['config']['lr'],
            'Batch Size': exp['config']['batch_size'],
            'Epochs': exp['config']['max_epochs'],
        }
        
        # Try to get metrics
        try:
            log_file = os.path.join(work_dir, "vis_data", "scalars.json")
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = [json.loads(line) for line in f]
                
                # Get best NME
                nme_values = [log['NME'] for log in logs if 'NME' in log and log.get('mode') == 'val']
                if nme_values:
                    result['Best NME'] = min(nme_values)
                    result['Final NME'] = nme_values[-1] if nme_values else 'N/A'
                
                # Get final training loss
                train_losses = [log['loss'] for log in logs if 'loss' in log and log.get('mode') == 'train']
                if train_losses:
                    result['Final Loss'] = train_losses[-1]
        except:
            pass
        
        results.append(result)
    
    # Convert to DataFrame for nice display
    df = pd.DataFrame(results)
    
    # Sort by Best NME if available
    if 'Best NME' in df.columns:
        df = df.sort_values('Best NME')
    
    print("\n" + "="*120)
    print("EXPERIMENT COMPARISON")
    print("="*120)
    print(df.to_string(index=False))
    print("="*120)
    
    # Print best experiment
    if 'Best NME' in df.columns:
        best_exp = df.iloc[0]
        print(f"\n🏆 BEST EXPERIMENT: {best_exp['Name']} (ID: {best_exp['ID']})")
        print(f"   Best NME: {best_exp['Best NME']:.4f}")

def get_experiment_details(index):
    """Get detailed information about a specific experiment."""
    if index < 0 or index >= len(experiments):
        print(f"Invalid experiment index: {index}")
        return
    
    exp = experiments[index]
    work_dir = f"work_dirs/experiment_{index}_{exp['name']}"
    
    print("\n" + "="*80)
    print(f"EXPERIMENT {index}: {exp['name']}")
    print("="*80)
    print(f"Description: {exp['description']}")
    print("\nConfiguration:")
    for key, value in exp['config'].items():
        print(f"  {key}: {value}")
    
    if os.path.exists(work_dir):
        print(f"\nWork Directory: {work_dir}")
        
        # Check status
        checkpoints = glob.glob(os.path.join(work_dir, "best_NME_epoch_*.pth"))
        if checkpoints:
            print(f"Status: Completed")
            print(f"Best Checkpoint: {os.path.basename(checkpoints[0])}")
        elif os.path.exists(os.path.join(work_dir, "experiment_error.txt")):
            print(f"Status: Failed")
            with open(os.path.join(work_dir, "experiment_error.txt"), 'r') as f:
                print("\nError Details:")
                print(f.read())
        else:
            print(f"Status: In Progress or Interrupted")
        
        # Try to show metrics
        try:
            log_file = os.path.join(work_dir, "vis_data", "scalars.json")
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = [json.loads(line) for line in f]
                
                nme_values = [log['NME'] for log in logs if 'NME' in log and log.get('mode') == 'val']
                if nme_values:
                    print(f"\nValidation Metrics:")
                    print(f"  Best NME: {min(nme_values):.4f}")
                    print(f"  Final NME: {nme_values[-1]:.4f}")
                    print(f"  Total Validations: {len(nme_values)}")
        except:
            pass
        
        # List files in work_dir
        print(f"\nFiles in work directory:")
        for file in os.listdir(work_dir):
            print(f"  - {file}")
    else:
        print(f"\nStatus: Not Started")
    
    print("="*80)

def clean_experiment(index):
    """Clean up experiment directory."""
    if index < 0 or index >= len(experiments):
        print(f"Invalid experiment index: {index}")
        return
    
    exp = experiments[index]
    work_dir = f"work_dirs/experiment_{index}_{exp['name']}"
    
    if os.path.exists(work_dir):
        import shutil
        response = input(f"Are you sure you want to delete {work_dir}? (y/N): ")
        if response.lower() == 'y':
            shutil.rmtree(work_dir)
            print(f"Deleted: {work_dir}")
    else:
        print(f"Experiment directory does not exist: {work_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Experiment management utilities')
    parser.add_argument('--list', action='store_true', help='List all experiments')
    parser.add_argument('--compare', nargs='*', type=int, help='Compare experiments (no args = all completed)')
    parser.add_argument('--details', type=int, help='Show details for specific experiment')
    parser.add_argument('--clean', type=int, help='Clean experiment directory')
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
    elif args.compare is not None:
        compare_experiments(args.compare if args.compare else None)
    elif args.details is not None:
        get_experiment_details(args.details)
    elif args.clean is not None:
        clean_experiment(args.clean)
    else:
        parser.print_help() 