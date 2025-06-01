#!/usr/bin/env python3
"""
Master script to run multiple cephalometric landmark detection experiments.
"""

import os
import subprocess
import shutil

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
BASE_WORK_DIR = "work_dirs"

def find_best_checkpoint(work_dir):
    """Finds the best_NME_epoch_*.pth checkpoint in a work_dir."""
    import glob
    pattern = os.path.join(work_dir, "best_NME_epoch_*.pth")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        pattern = os.path.join(work_dir, "epoch_*.pth") # Fallback to any epoch
        checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getctime)

def update_config_load_from(config_path, new_load_from_path):
    """Reads a config, updates load_from, and writes it back."""
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    updated_lines = []
    found_load_from = False
    for line in lines:
        if line.strip().startswith('load_from'):
            updated_lines.append(f"load_from = '{new_load_from_path}'\n")
            found_load_from = True
        else:
            updated_lines.append(line)
    
    if not found_load_from:
        # Add load_from if not present (e.g. if it was None in base)
        # Ensure it's added in a sensible place, e.g., after _base_
        idx_to_insert = 0
        for i, line in enumerate(updated_lines):
            if line.strip().startswith('_base_'):
                idx_to_insert = i + 1
                break
        updated_lines.insert(idx_to_insert, f"load_from = '{new_load_from_path}'\n")
        
    with open(config_path, 'w') as f:
        f.writelines(updated_lines)
    print(f"Updated '{config_path}' to load from '{new_load_from_path}'")

def main():
    completed_work_dirs = []

    for i, (config_file, work_dir_suffix, load_from_exp_idx) in enumerate(experiments):
        print("\n" + "="*80)
        print(f"🚀 STARTING EXPERIMENT {i+1}/{len(experiments)}: {work_dir_suffix}")
        print(f"   Config: {config_file}")
        print("="*80)

        # Construct full work_dir path
        current_work_dir = os.path.join(BASE_WORK_DIR, work_dir_suffix)
        
        # Prepare config for this run (especially for multi-stage experiments)
        temp_config_file = f"temp_config_exp{i+1}.py"
        shutil.copy(config_file, temp_config_file)

        # Handle loading from a previous experiment's checkpoint for multi-stage
        if load_from_exp_idx is not None:
            if load_from_exp_idx >= len(completed_work_dirs):
                print(f"ERROR: Experiment {i+1} depends on Exp {load_from_exp_idx+1}, which has not run or failed.")
                continue
            
            prev_exp_work_dir = completed_work_dirs[load_from_exp_idx]
            checkpoint_to_load = find_best_checkpoint(prev_exp_work_dir)
            
            if not checkpoint_to_load:
                print(f"ERROR: No checkpoint found in {prev_exp_work_dir} to load for Exp {i+1}.")
                # Clean up temp config before skipping
                if os.path.exists(temp_config_file):
                    os.remove(temp_config_file)
                continue
            
            print(f"   Updating config to load from: {checkpoint_to_load}")
            update_config_load_from(temp_config_file, checkpoint_to_load)
        else:
            # For single-stage experiments, ensure `load_from` in their config is used (or default AFLW if specified)
            # If their config has `load_from = None`, it will use the base model's init_cfg.
            # If `load_from` is set in `expX_...py`, it will use that.
            # Our exp2_stage1 already sets its own specific `load_from`.
            pass 

        # Command to run the training script
        # The training script (train_improved_v4.py) needs to:
        # 1. Accept the config file path.
        # 2. Use the `work_dir` specified in the config file itself (after being set by runner.py from_cfg).
        #    The training script itself sets cfg.work_dir based on its hardcoded `work_dir` which then gets overridden by the specific experiment config.
        #    Alternatively, the training script could take --work-dir as cmd arg.
        #    For this master script, we will rely on `train_improved_v4.py` to set its own experiment-specific `work_dir` string,
        #    and then its `cfg.work_dir` will be correctly set by the loaded config.

        # To ensure each experiment uses its own work_dir, the master script should pass it.
        # We need to modify train_improved_v4.py to accept --work-dir and use it.
        # OR, more simply, ensure each experiment config defines its specific work_dir. 
        # The latter is cleaner.
        
        # Let's assume train_improved_v4.py is modified to accept --config and --work-dir
        # This is the ideal way. For now, I will create a copy of train_improved_v4.py for each experiment
        # and modify its work_dir and config_path to avoid complex cmd line parsing in train_improved_v4.py

        # Create a temporary training script for this experiment
        temp_train_script = f"temp_train_exp{i+1}.py"
        shutil.copy(TRAINING_SCRIPT, temp_train_script)

        # Modify the temp training script to use the correct config and work_dir
        with open(temp_train_script, 'r') as f:
            script_content = f.read()
        
        # Replace config_path and work_dir in the temp script
        # Ensure these placeholder strings are unique enough not to cause issues.
        # Example: In train_improved_v4.py, have lines like:
        # CONFIG_PATH_PLACEHOLDER = "Pretrained_model/hrnetv2_w18_cephalometric_256x256_finetune.py"
        # WORK_DIR_PLACEHOLDER = "work_dirs/hrnetv2_w18_cephalometric_384x384_ohkm_v4"
        # And replace these.
        # For now, assuming train_improved_v4.py is flexible or we modify fixed lines.
        
        # Simplified approach: The train_improved_v4.py loads its config and that config specifies the work_dir.
        # So, we just need to ensure train_improved_v4.py loads the *correct experiment config*.
        # The `work_dir` inside `train_improved_v4.py` will be overridden by the loaded experiment config.

        # Let's just call the original training script with the temp_config_file
        # The train_improved_v4.py script needs to be modified to take the config as an argument.
        # For now, I will assume you will modify train_improved_v4.py to take config path as sys.argv[1]
        # and it will use the work_dir specified *inside* that config file.

        cmd = ["python", temp_train_script, temp_config_file] 
        # (This requires train_improved_v4.py to be adaptable)
        # A better way: modify train_improved_v4.py to take --config and --work-dir
        # Then cmd = ["python", TRAINING_SCRIPT, "--config", temp_config_file, "--work-dir", current_work_dir]
        
        # Simpler: Assume train_improved_v4.py is modified to set config_path and work_dir
        # based on the experiment config it loads. We will make a copy and modify its internal paths.

        lines = []
        with open(temp_train_script, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line_num, line_content in enumerate(lines):
            if "config_path = " in line_content and "# Relative path" not in line_content and "# MODIFIED" not in line_content and "# User might be in Colab" not in line_content:
                new_lines.append(f'    config_path = "{temp_config_file}"\n') 
            elif "work_dir = " in line_content and "# Relative path" not in line_content and "# MODIFIED" not in line_content and "# User might be in Colab" not in line_content:
                new_lines.append(f'    work_dir = "{current_work_dir}"\n')
            else:
                new_lines.append(line_content)
        
        with open(temp_train_script, 'w') as f:
            f.writelines(new_lines)

        print(f"   Executing: python {temp_train_script}")
        try:
            process = subprocess.Popen(["python", temp_train_script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            rc = process.poll()
            if rc == 0:
                print(f"✅ EXPERIMENT {i+1} ({work_dir_suffix}) COMPLETED SUCCESSFULLY.")
                completed_work_dirs.append(current_work_dir)
            else:
                print(f"❌ EXPERIMENT {i+1} ({work_dir_suffix}) FAILED with exit code {rc}.")
                completed_work_dirs.append(None) # Placeholder for failed experiment
        except Exception as e:
            print(f"❌ EXPERIMENT {i+1} ({work_dir_suffix}) CRASHED: {e}")
            completed_work_dirs.append(None)
        finally:
            # Clean up temporary files
            if os.path.exists(temp_config_file):
                os.remove(temp_config_file)
            if os.path.exists(temp_train_script):
                os.remove(temp_train_script)

    print("\n" + "="*80)
    print("🏁 ALL EXPERIMENTS FINISHED 🏁")
    print("="*80)
    print("Results saved in respective subdirectories under work_dirs/")
    print("Completed work directories (or None if failed):")
    for wd in completed_work_dirs:
        print(f"  - {wd}")

if __name__ == "__main__":
    main() 