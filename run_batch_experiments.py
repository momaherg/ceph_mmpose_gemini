#!/usr/bin/env python3
"""
Batch runner for multiple experiments.
Usage: python run_batch_experiments.py --experiments 0 2 3 --sequential
"""

import os
import subprocess
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from experiments_config import experiments

def run_single_experiment(index):
    """Run a single experiment and return the result."""
    exp_name = experiments[index]['name']
    print(f"\n{'='*80}")
    print(f"Starting Experiment {index}: {exp_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Run the experiment
        result = subprocess.run(
            ['python', 'run_experiment.py', '--index', str(index)],
            capture_output=True,
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ Experiment {index} completed in {elapsed/60:.1f} minutes")
        
        return {
            'index': index,
            'name': exp_name,
            'status': 'completed',
            'time': elapsed,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Experiment {index} failed after {elapsed/60:.1f} minutes")
        
        return {
            'index': index,
            'name': exp_name,
            'status': 'failed',
            'time': elapsed,
            'stdout': e.stdout,
            'stderr': e.stderr,
            'error': str(e)
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Experiment {index} failed with exception: {e}")
        
        return {
            'index': index,
            'name': exp_name,
            'status': 'error',
            'time': elapsed,
            'error': str(e)
        }

def run_experiments_sequential(experiment_indices):
    """Run experiments sequentially."""
    results = []
    
    for idx in experiment_indices:
        result = run_single_experiment(idx)
        results.append(result)
        
        # Save intermediate results
        save_batch_results(results, 'sequential')
    
    return results

def run_experiments_parallel(experiment_indices, max_workers=2):
    """Run experiments in parallel."""
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments
        future_to_idx = {executor.submit(run_single_experiment, idx): idx 
                        for idx in experiment_indices}
        
        # Process completed experiments
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results.append(result)
                print(f"\n📊 Completed {len(results)}/{len(experiment_indices)} experiments")
                
                # Save intermediate results
                save_batch_results(results, 'parallel')
            except Exception as e:
                print(f"\n❌ Experiment {idx} raised exception: {e}")
                results.append({
                    'index': idx,
                    'name': experiments[idx]['name'],
                    'status': 'exception',
                    'error': str(e)
                })
    
    return results

def save_batch_results(results, mode):
    """Save batch run results to file."""
    import json
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"batch_results_{mode}_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")

def print_summary(results, total_time):
    """Print summary of batch run."""
    print("\n" + "="*80)
    print("BATCH RUN SUMMARY")
    print("="*80)
    
    completed = [r for r in results if r['status'] == 'completed']
    failed = [r for r in results if r['status'] in ['failed', 'error', 'exception']]
    
    print(f"Total experiments: {len(results)}")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_time/60:.1f} minutes")
    
    if completed:
        print("\n✅ Completed experiments:")
        for r in completed:
            print(f"   - {r['index']}: {r['name']} ({r['time']/60:.1f} min)")
    
    if failed:
        print("\n❌ Failed experiments:")
        for r in failed:
            print(f"   - {r['index']}: {r['name']} ({r.get('error', 'Unknown error')})")
    
    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description='Run multiple experiments in batch')
    parser.add_argument('--experiments', nargs='+', type=int, required=True,
                       help='List of experiment indices to run')
    parser.add_argument('--sequential', action='store_true',
                       help='Run experiments sequentially (default is parallel)')
    parser.add_argument('--max-workers', type=int, default=2,
                       help='Maximum parallel workers (default: 2)')
    parser.add_argument('--skip-completed', action='store_true',
                       help='Skip experiments that have already been completed')
    
    args = parser.parse_args()
    
    # Validate experiment indices
    invalid_indices = [idx for idx in args.experiments 
                       if idx < 0 or idx >= len(experiments)]
    if invalid_indices:
        print(f"Error: Invalid experiment indices: {invalid_indices}")
        print(f"Valid range: 0-{len(experiments)-1}")
        return
    
    # Filter experiments if skip_completed is set
    experiment_indices = args.experiments
    if args.skip_completed:
        to_run = []
        for idx in experiment_indices:
            work_dir = f"work_dirs/experiment_{idx}_{experiments[idx]['name']}"
            import glob
            if not (os.path.exists(work_dir) and 
                   glob.glob(os.path.join(work_dir, "best_NME_epoch_*.pth"))):
                to_run.append(idx)
            else:
                print(f"Skipping completed experiment {idx}: {experiments[idx]['name']}")
        experiment_indices = to_run
    
    if not experiment_indices:
        print("No experiments to run.")
        return
    
    # Print experiments to run
    print("\n" + "="*80)
    print("EXPERIMENTS TO RUN")
    print("="*80)
    for idx in experiment_indices:
        print(f"{idx}: {experiments[idx]['name']} - {experiments[idx]['description']}")
    print("="*80)
    
    # Confirm
    response = input(f"\nRun {len(experiment_indices)} experiments? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Run experiments
    start_time = time.time()
    
    if args.sequential:
        print("\n🔄 Running experiments sequentially...")
        results = run_experiments_sequential(experiment_indices)
    else:
        print(f"\n🔄 Running experiments in parallel (max workers: {args.max_workers})...")
        results = run_experiments_parallel(experiment_indices, args.max_workers)
    
    total_time = time.time() - start_time
    
    # Print summary
    print_summary(results, total_time)
    
    # Final comparison
    print("\n📊 To compare all results, run:")
    print("   python experiment_utils.py --compare")

if __name__ == "__main__":
    main() 