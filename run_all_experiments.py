#!/usr/bin/env python3
"""
Master script to run multiple cephalometric landmark detection experiments.
"""

import os
import subprocess
import shutil
import argparse
import glob

# Define experiments: (config_file, work_dir_suffix, load_from_checkpoint_from_exp_idx)
# `load_from_checkpoint_from_exp_idx` = None to use config's default `load_from`.
# Otherwise, it's the 0-indexed experiment number to load the best checkpoint from.
experiments = [
    ("Pretrained_model/exp1_focal_ohkm.py", "exp1_focal_ohkm", None),
    ("Pretrained_model/exp2_stage1_256_ohkm.py", "exp2_stage1_256_ohkm", None),
    ("Pretrained_model/exp2_stage2_384_adaptive.py", "exp2_stage2_384_adaptive", 1), # Loads from exp2_stage1
    ("Pretrained_model/exp3_mlecc.py", "exp3_mlecc", None),
    ("Pretrained_model/exp4_augmentations.py", "exp4_augmentations", None),
]

# Path to your main training script (train_improved_v4.py or similar)
# Make sure this script can accept --work-dir and --cfg-options for load_from if needed.
# For simplicity, we will modify the config directly for load_from in Exp2 Stage2.
TRAINING_SCRIPT = "train_improved_v4.py" # Or your V4 training script
# BASE_WORK_DIR will now be controlled by args.work_dir_root

def find_best_checkpoint(work_dir):
    """Finds the best_NME_epoch_*.pth checkpoint in a work_dir."""
    pattern = os.path.join(work_dir, "best_NME_epoch_*.pth")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        pattern = os.path.join(work_dir, "epoch_*.pth") # Fallback to any epoch
        checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

def update_config_load_from(config_path, new_load_from_path):
    """Reads a config, updates load_from, and writes it back."""
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    updated_lines = []
    found_load_from = False
    for line in lines:
        if line.strip().startswith('load_from'):
            updated_lines.append(f"load_from = '{new_load_from_path}'\\n")
            found_load_from = True
        else:
            updated_lines.append(line)
    
    if not found_load_from:
        idx_to_insert = 0
        for i, line_content in enumerate(updated_lines):
            if line_content.strip().startswith('_base_'):
                idx_to_insert = i + 1
                break
        updated_lines.insert(idx_to_insert, f"load_from = '{new_load_from_path}'\\n")
        
    with open(config_path, 'w') as f:
        f.writelines(updated_lines)
    print(f"Updated '{config_path}' to load from '{new_load_from_path}'")

def main():
    parser = argparse.ArgumentParser(description="Run Cephalometric Landmark Detection Experiments.")
    parser.add_argument(
        "--experiment-index",
        type=int,
        default=None,
        help="0-indexed number of the experiment to run. If not provided, all experiments are run sequentially."
    )
    parser.add_argument(
        "--work-dir-root",
        type=str,
        default="work_dirs",
        help="Root directory for experiment outputs and for finding dependency checkpoints (default: 'work_dirs')."
    )
    parser.add_argument(
        "--dependency-checkpoint-path",
        type=str,
        default=None,
        help="Path to a specific checkpoint to load if the selected experiment has a dependency. Overrides automatic search. Only used if --experiment-index is specified."
    )
    args = parser.parse_args()

    BASE_WORK_DIR = args.work_dir_root

    experiments_to_process = []
    if args.experiment_index is not None:
        if not (0 <= args.experiment_index < len(experiments)):
            print(f"ERROR: --experiment-index {args.experiment_index} is out of range (0-{len(experiments)-1}).")
            return
        # Store as (original_index, experiment_details)
        experiments_to_process.append((args.experiment_index, experiments[args.experiment_index]))
        print(f"Targeting Experiment {args.experiment_index + 1}: {experiments[args.experiment_index][1]}")
    else:
        experiments_to_process = list(enumerate(experiments))
        print("Running all experiments sequentially.")

    completed_work_dirs_map = {} # Stores {index: work_dir_path} for successful runs

    for original_idx, experiment_details in experiments_to_process:
        config_file, work_dir_suffix, load_from_exp_idx = experiment_details
        
        print("\n" + "="*80)
        print(f"🚀 STARTING EXPERIMENT {original_idx+1}/{len(experiments)}: {work_dir_suffix}")
        print(f"   Config: {config_file}")
        print("="*80)

        current_work_dir = os.path.join(BASE_WORK_DIR, work_dir_suffix)
        if not os.path.exists(current_work_dir):
            os.makedirs(current_work_dir, exist_ok=True)
        
        temp_config_file_name = f"temp_config_exp{original_idx+1}_{work_dir_suffix}.py"
        temp_config_file = os.path.join(current_work_dir, temp_config_file_name) # Place temp config in work_dir
        shutil.copy(config_file, temp_config_file)

        checkpoint_to_load_for_config = None

        if load_from_exp_idx is not None:
            if args.experiment_index is not None: # Single experiment mode with this specific experiment selected
                if args.dependency_checkpoint_path:
                    checkpoint_to_load_for_config = args.dependency_checkpoint_path
                    if not os.path.exists(checkpoint_to_load_for_config):
                        print(f"ERROR: Manually specified dependency checkpoint not found: {checkpoint_to_load_for_config}")
                        if os.path.exists(temp_config_file): os.remove(temp_config_file)
                        continue # or return if fatal for single run
                    print(f"   Using manually specified dependency checkpoint: {checkpoint_to_load_for_config}")
                else:
                    # Auto-search for dependency checkpoint
                    dependent_exp_details = experiments[load_from_exp_idx]
                    dependent_exp_work_dir_suffix = dependent_exp_details[1]
                    # prev_exp_work_dir is where the output of the dependency should be
                    prev_exp_work_dir = os.path.join(BASE_WORK_DIR, dependent_exp_work_dir_suffix)
                    print(f"   Searching for checkpoint in presumed dependency directory: {prev_exp_work_dir}")
                    checkpoint_to_load_for_config = find_best_checkpoint(prev_exp_work_dir)
                    if checkpoint_to_load_for_config:
                        print(f"   Found dependency checkpoint: {checkpoint_to_load_for_config}")
                    else:
                        print(f"ERROR: No checkpoint found in '{prev_exp_work_dir}' to load for Exp {original_idx+1} (depends on Exp {load_from_exp_idx+1}).")
                        print(f"       Please ensure Exp {load_from_exp_idx+1} has run successfully and its output is in '{prev_exp_work_dir}',")
                        print(f"       or provide the checkpoint path using --dependency-checkpoint-path.")
                        if os.path.exists(temp_config_file): os.remove(temp_config_file)
                        continue # or return
            else: # Running all experiments sequentially
                if load_from_exp_idx not in completed_work_dirs_map:
                    print(f"ERROR: Experiment {original_idx+1} ({work_dir_suffix}) depends on Exp {load_from_exp_idx+1}, which has not completed successfully or was skipped.")
                    if os.path.exists(temp_config_file): os.remove(temp_config_file)
                    continue
                
                prev_exp_work_dir = completed_work_dirs_map[load_from_exp_idx]
                checkpoint_to_load_for_config = find_best_checkpoint(prev_exp_work_dir)
                
                if not checkpoint_to_load_for_config:
                    print(f"ERROR: No checkpoint found in {prev_exp_work_dir} (from completed Exp {load_from_exp_idx+1}) to load for Exp {original_idx+1}.")
                    if os.path.exists(temp_config_file): os.remove(temp_config_file)
                    continue
                print(f"   Found checkpoint from previous successful run (Exp {load_from_exp_idx+1}): {checkpoint_to_load_for_config}")

        if checkpoint_to_load_for_config:
            update_config_load_from(temp_config_file, checkpoint_to_load_for_config)
        else:
            # If load_from_exp_idx was None, or if it was set but no checkpoint found (error handled above),
            # the temp_config_file (copy of original) will use its own `load_from` or mmpose default.
            # This is correct for experiments like exp2_stage1 which specify their own non-dependent load_from.
            if load_from_exp_idx is not None: # This means we expected a checkpoint but didn't set one (due to error)
                 print(f"   Warning: Proceeding for Exp {original_idx+1} without an explicit loaded checkpoint, despite dependency. Original config's load_from will be used.")
            else: # load_from_exp_idx is None
                 print(f"   Exp {original_idx+1} does not have a listed inter-experiment dependency for checkpoint loading. Original config's 'load_from' will be used if set.")


        temp_train_script_name = f"temp_train_exp{original_idx+1}_{work_dir_suffix}.py"
        temp_train_script = os.path.join(current_work_dir, temp_train_script_name) # Place temp script in work_dir
        shutil.copy(TRAINING_SCRIPT, temp_train_script)

        lines = []
        with open(temp_train_script, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        # This modification method is fragile. Ideally, train_improved_v4.py would take args.
        for line_content in lines:
            # Ensure we are replacing the intended lines, possibly by making them more unique in train_improved_v4.py
            # Example placeholders in train_improved_v4.py:
            #   config_path = "CONFIG_PATH_PLACEHOLDER"
            #   work_dir = "WORK_DIR_PLACEHOLDER"
            if "config_path = " in line_content and "PLACEHOLDER_CONFIG_PATH" not in line_content and "# MODIFIED" not in line_content and "# User might be in Colab" not in line_content and "checkpoint_config" not in line_content and not line_content.strip().startswith("#"):
                # Assuming a line like: config_path = "some/default/path.py"
                indent = line_content[:line_content.find("config_path = ")]
                new_lines.append(f'{indent}config_path = "{os.path.abspath(temp_config_file)}" # MODIFIED BY RUNNER\\n')
            elif "work_dir = " in line_content and "PLACEHOLDER_WORK_DIR" not in line_content and "# MODIFIED" not in line_content and "# User might be in Colab" not in line_content and not line_content.strip().startswith("#"):
                # Assuming a line like: work_dir = "some/default/work_dir"
                indent = line_content[:line_content.find("work_dir = ")]
                new_lines.append(f'{indent}work_dir = "{os.path.abspath(current_work_dir)}" # MODIFIED BY RUNNER\\n')
            else:
                new_lines.append(line_content)
        
        with open(temp_train_script, 'w') as f:
            f.writelines(new_lines)

        print(f"   Executing: python {temp_train_script}")
        print(f"   Outputting to: {current_work_dir}")
        print(f"   Using config: {temp_config_file}")
        try:
            # Run from the directory of the temp script to handle relative paths if any in train script
            process = subprocess.Popen(["python", os.path.basename(temp_train_script)], cwd=os.path.dirname(temp_train_script), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            rc = process.poll()
            if rc == 0:
                print(f"✅ EXPERIMENT {original_idx+1} ({work_dir_suffix}) COMPLETED SUCCESSFULLY.")
                completed_work_dirs_map[original_idx] = current_work_dir
            else:
                print(f"❌ EXPERIMENT {original_idx+1} ({work_dir_suffix}) FAILED with exit code {rc}.")
                # No entry in completed_work_dirs_map for failed experiments
        except Exception as e:
            print(f"❌ EXPERIMENT {original_idx+1} ({work_dir_suffix}) CRASHED: {e}")
        finally:
            # Optionally, clean up temporary files, or leave them for debugging
            # if os.path.exists(temp_config_file):
            #     os.remove(temp_config_file)
            # if os.path.exists(temp_train_script):
            #     os.remove(temp_train_script)
            pass


    print("\n" + "="*80)
    print("🏁 ALL PROCESSING FINISHED 🏁")
    print("="*80)
    if args.experiment_index is None:
        print("Results saved in respective subdirectories under work_dirs/")
        print("Successfully completed experiment work directories:")
        for idx, wd in completed_work_dirs_map.items():
            print(f"  - Exp {idx+1} ({experiments[idx][1]}): {wd}")
    else:
        print(f"Experiment {args.experiment_index+1} processing complete. Results in: {current_work_dir if 'current_work_dir' in locals() else 'N/A - Check logs'}")


if __name__ == "__main__":
    main() 