import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner, TestLoop
from mmengine.registry import init_default_scope

# Initialize the scope BEFORE importing custom modules that register components
print("Initializing mmpose scope before imports...")
init_default_scope('mmpose')

# Import your custom modules to ensure they are registered
# These imports are crucial for the MMEngine registry to find your custom classes
print("Importing custom modules for registration...")
import custom_cephalometric_dataset # Registers CustomCephalometricDataset
import custom_transforms           # Registers LoadImageNumpy
import cephalometric_dataset_info  # Makes dataset_info available for the config file
import mre_metric                  # Registers MeanRadialError metric

def parse_args():
    parser = argparse.ArgumentParser(description='Train or Test a pose model using MMEngine')
    parser.add_argument(
        'config',
        default='configs/hrnetv2/hrnetv2_w18_cephalometric_224x224.py',
        help='train/test config file path'
    )
    parser.add_argument('--work-dir', help='the directory to save logs and models')
    parser.add_argument('--checkpoint', help='checkpoint file to load for testing')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair ' 
        'in xxx=yyy format will be merged into config file. If the value to ' 
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b ' 
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]". ' 
        'Note that string values are surrounded with "". Example: ' 
        'python main.py configs/hrnetv2/... --cfg-options model.backbone.depth=18'
    )
    parser.add_argument(
        '--mode', 
        choices=['train', 'test'], 
        default='train', 
        help='Whether to run training or testing'
    )
    args = parser.parse_args() # Use this for script execution
    # For Jupyter, you might need to provide args manually if not running from CLI:
    # args = parser.parse_args(args=[] + ['--mode', 'test', '--checkpoint', 'path/to/your.pth'])
    return args

def evaluate_checkpoint(config_path: str, checkpoint_path: str, cfg_options: dict = None):
    """Evaluate a model checkpoint with the MRE metric."""
    print(f"\n--- Starting Evaluation for Checkpoint: {checkpoint_path} ---")
    
    # --- Configuration Loading for Evaluation ---
    cfg = Config.fromfile(config_path)
    if cfg_options:
        cfg.merge_from_dict(cfg_options)

    # --- Setup for Testing --- 
    test_ann_file = 'dev_data_pure_old_numpy.json' # Or your specific test set JSON
    
    if not osp.exists(osp.join(cfg.data_root, test_ann_file)):
        print(f"Warning: Test annotation file '{test_ann_file}' not found at '{osp.join(cfg.data_root, test_ann_file)}'. Using train set for evaluation as a fallback.")
        print("This is NOT recommended for proper evaluation. Please provide a valid test/dev set.")
        test_ann_file = cfg.train_dataloader.dataset.ann_file # Fallback to train set if dev set not found

    cfg.test_dataloader = dict(
        batch_size=32,
        num_workers=4,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type=cfg.dataset_type,
            data_root=cfg.data_root,
            ann_file=test_ann_file, 
            metainfo=cfg.train_dataloader.dataset.metainfo,
            pipeline=cfg.val_pipeline if hasattr(cfg, 'val_pipeline') and cfg.val_pipeline else cfg.train_pipeline,
            test_mode=True,
        )
    )
    cfg.test_evaluator = dict(type='MeanRadialError')
    cfg.test_cfg = dict()

    # --- Work Directory ---
    cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config_path))[0] + '_test')
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    print(f"Test work directory set to: {osp.abspath(cfg.work_dir)}")

    # --- Build the Runner ---
    runner = Runner.from_cfg(cfg)
    
    # --- Disable flip testing ---
    # Cephalometric landmarks don't have clear left-right counterparts
    # This is needed because model's test_cfg tries to use flip_indices
    print("Disabling flip testing for cephalometric landmarks...")
    if hasattr(runner.model, 'test_cfg'):
        runner.model.test_cfg.flip_test = False
    
    # --- Load Checkpoint ---
    print(f"Loading checkpoint '{checkpoint_path}'...")
    import torch
    try:
        # Use torch.load directly with weights_only=False
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            runner.model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("Checkpoint loaded successfully with weights_only=False.")
        else:
            print("Warning: Loaded checkpoint doesn't contain 'state_dict'. Trying to load directly...")
            runner.model.load_state_dict(checkpoint, strict=False)
            print("Checkpoint loaded directly as a state dict.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Evaluation will continue with initialized weights.")

    # --- Start Testing/Evaluation ---
    try:
        print("Launching evaluation...")
        metrics = runner.test()
        print("--- Evaluation Finished ---")
        print("Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        return metrics
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Scope is already initialized at the top of the file
    args = parse_args()

    print(f"Running in mode: {args.mode}")

    # --- Configuration Loading ---
    config_path = args.config
    cfg_options = args.cfg_options
    
    if args.mode == 'test':
        if not args.checkpoint:
            print("Error: Checkpoint path must be provided for testing. Use --checkpoint <path_to_checkpoint>")
            return
        evaluate_checkpoint(config_path, args.checkpoint, cfg_options)
        return # Exit after testing

    # --- Training Mode (existing code) --- 
    print("Starting training process...")
    work_dir = args.work_dir

    if not osp.exists(config_path):
        print(f"Config file not found at: {config_path}")
        print("Please ensure the config file exists at the specified path.")
        print(f"Current CWD: {os.getcwd()}")
        if osp.exists(osp.join(os.getcwd(), config_path)):
            print(f"Found config at: {osp.join(os.getcwd(), config_path)}")
        return

    cfg = Config.fromfile(config_path)
    if cfg_options:
        cfg.merge_from_dict(cfg_options)
    
    if work_dir is not None:
        cfg.work_dir = work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config_path))[0])
    
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    print(f"Work directory set to: {osp.abspath(cfg.work_dir)}")

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
    print("Runner built successfully.")

    try:
        print("Launching training...")
        runner.train()
        print("Training finished.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == '__main__':
    main()
