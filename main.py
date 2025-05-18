import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

# Import your custom modules to ensure they are registered
# These imports are crucial for the MMEngine registry to find your custom classes
import custom_cephalometric_dataset # Registers CustomCephalometricDataset
import custom_transforms           # Registers LoadImageNumpy
import cephalometric_dataset_info  # Makes dataset_info available for the config file

# --- Diagnostic print --- #
from mmpose.registry import TRANSFORMS as MMPTR_DIAGNOSTIC
if 'LoadImageNumpy' in MMPTR_DIAGNOSTIC.module_dict:
    print("DIAGNOSTIC: 'LoadImageNumpy' IS found in mmpose.registry.TRANSFORMS after import.")
else:
    print("DIAGNOSTIC: 'LoadImageNumpy' IS NOT found in mmpose.registry.TRANSFORMS after import. Registration failed or is not visible.")
# --- End Diagnostic print --- #

from mmengine.registry import init_default_scope

def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose model using MMEngine')
    parser.add_argument(
        'config',
        default='configs/hrnetv2/hrnetv2_w18_cephalometric_224x224.py',
        help='train config file path'
    )
    parser.add_argument('--work-dir', help='the directory to save logs and models')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair ' 
        'in xxx=yyy format will be merged into config file. If the value to ' 
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b ' 
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]". '
        'Note that string values are surrounded with "". Example: ' 
        'python tools/train.py configs/hrnetv2_w18_cephalometric_224x224.py ' 
        '--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'
    )
    # When running in Jupyter, args might be an empty list or None
    # Provide a default for args if running in such an environment
    # args = parser.parse_args(args=[]) # For Jupyter, pass empty list
    # However, for a script, we expect command-line args or direct specification.
    # For this setup, we'll use the default config path directly.
    # args = parser.parse_args([]) # Use this if you need to run parse_args() in a notebook cell with no CLI args
    return parser

def main():
    # init_default_scope('mmpose') # Initialize the mmpose scope to load mmpose components
    # The config does this implicitly if it uses mmpose types, or you can do it explicitly.
    # It's good practice to have it if you're using types from mmpose registry directly.
    # If your custom dataset/transforms are in a different scope, initialize that one.
    init_default_scope('mmpose') # Or your project's scope if you have one

    print("Starting training process...")
    
    # --- Configuration Loading ---
    # Use a fixed config path for simplicity in this notebook-like environment
    # In a typical script, you'd use parse_args()
    config_path = 'configs/hrnetv2/hrnetv2_w18_cephalometric_224x224.py'
    work_dir = 'work_dirs/hrnetv2_w18_cephalometric_experiment' # Define a working directory
    cfg_options = None # No overrides for now

    if not osp.exists(config_path):
        print(f"Config file not found at: {config_path}")
        print("Please ensure the config file exists at the specified path.")
        print(f"Current CWD: {os.getcwd()}")
        # Try to provide a hint if it's in a common relative location
        if osp.exists(osp.join(os.getcwd(), config_path)):
            print(f"Found config at: {osp.join(os.getcwd(), config_path)}")
        return

    cfg = Config.fromfile(config_path)

    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    
    # --- Setup Work Directory ---
    if work_dir is not None:
        cfg.work_dir = work_dir
    elif cfg.get('work_dir', None) is None:
        # Use config name as work_dir if not set
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config_path))[0])
    
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    print(f"Work directory set to: {osp.abspath(cfg.work_dir)}")

    # --- Build the Runner ---
    # The runner is responsible for managing the training, validation, and testing loops.
    if 'runner_type' not in cfg:
        # Build a runner from config and registered modules
        runner = Runner.from_cfg(cfg)
    else:
        # Build a runner from registry
        runner = RUNNERS.build(cfg)
    
    print("Runner built successfully.")

    # --- Start Training ---
    # `train_dataloader` and `model` will be built by the runner based on the config
    try:
        print("Launching training...")
        runner.train()
        print("Training finished.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- (Optional) Start Testing/Evaluation after training ---
    # You might want to load the best checkpoint and evaluate
    # best_checkpoint_path = osp.join(cfg.work_dir, 'best_PCKAccuracy_epoch_X.pth') # Replace X with actual epoch
    # if osp.exists(best_checkpoint_path):
    #     print(f"\nStarting evaluation with best checkpoint: {best_checkpoint_path}")
    #     runner.load_checkpoint(best_checkpoint_path)
    #     val_dataloader_cfg = cfg.val_dataloader
    #     val_evaluator_cfg = cfg.val_evaluator
    #     metrics = runner.test(val_dataloader_cfg, val_evaluator_cfg)
    #     print("Evaluation metrics:", metrics)
    # else:
    #     print(f"Best checkpoint not found for evaluation. Looked in {cfg.work_dir}")

if __name__ == '__main__':
    # args = parse_args() # Uncomment if you want to use command-line arguments
    main()
