import os
import os.path as osp
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import init_default_scope
import torch
from mmengine.registry import RUNNERS

# These imports are crucial for the MMEngine registry to find your custom classes
# and for the config to load dataset metainfo.
# Ensure these files are in the same directory or Python path when running.
print("Initializing mmpose scope and importing custom modules for evaluation...")
init_default_scope('mmpose')
import custom_cephalometric_dataset
import custom_transforms
from cephalometric_dataset_info import dataset_info as cephalometric_metainfo # For explicit use if needed

# Define a custom runner class that handles the checkpoint loading issue
@RUNNERS.register_module(force=True)
class CustomRunner(Runner):
    """Custom Runner that overrides load_checkpoint to handle PyTorch 2.6+ loading restrictions."""
    
    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        """Load checkpoint with weights_only=False to handle MMEngine config objects."""
        print(f"Loading checkpoint from {filename} with weights_only=False for PyTorch 2.6+ compatibility...")
        try:
            # Try loading with weights_only=False for PyTorch 2.6+
            checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
            self.model.load_state_dict(checkpoint['state_dict'], strict=strict)
            return checkpoint
        except TypeError as e:
            # If weights_only is not a valid parameter (older PyTorch), try without it
            if 'weights_only' in str(e):
                print("Your PyTorch version doesn't have the weights_only parameter. Using default loading.")
                checkpoint = torch.load(filename, map_location=map_location)
                self.model.load_state_dict(checkpoint['state_dict'], strict=strict)
                return checkpoint
            else:
                raise

def evaluate_checkpoint_mre(config_file_path: str,
                            checkpoint_file_path: str,
                            test_ann_file_path: str,
                            data_root: str,
                            work_dir: str = 'temp_eval_work_dir'):
    """
    Evaluates a model checkpoint on a test set using Mean Radial Error (EPE).

    Args:
        config_file_path (str): Path to the MMPose model configuration file.
        checkpoint_file_path (str): Path to the saved model checkpoint (.pth file).
        test_ann_file_path (str): Path to the annotation file for the test set 
                                   (relative to data_root).
        data_root (str): The root directory for the dataset.
        work_dir (str): Directory to save temporary logs and files during evaluation.
    """

    print(f"Loading configuration from: {config_file_path}")
    cfg = Config.fromfile(config_file_path)

    # --- Override/Set up Test Dataloader Configuration ---
    print(f"Setting up test dataloader with ann_file: {test_ann_file_path} in data_root: {data_root}")
    
    # Extract necessary parts from the original config if they exist, or define them
    # These would ideally be fully defined in the original config's test_dataloader section
    # If they were None, we need to construct them.
    dataset_type = cfg.get('dataset_type', 'CustomCephalometricDataset')
    input_size = cfg.get('input_size', (224, 224)) # Get from global scope or model if not top-level
    if isinstance(input_size, dict): # Sometimes it might be within model config
        input_size = cfg.model.head.decoder.get('input_size', (224,224))


    # Define the test pipeline (should match val_pipeline if it was for testing)
    # Reconstructing based on what was in the original config's val_pipeline
    test_pipeline = [
        dict(type='LoadImageNumpy'),
        dict(type='GetBBoxCenterScale'),
        dict(type='TopdownAffine', input_size=input_size),
        dict(type='PackPoseInputs', 
             meta_keys=(
                'img_id', 'img_path', 'ori_shape', 'img_shape',
                'input_size', 'input_center', 'input_scale',
                'flip', 'flip_direction',
                'num_joints', 'joint_weights',
                'id', 'patient_text_id', 'set', 'class' # Custom keys
                )
            )
    ]

    cfg.test_dataloader = dict(
        batch_size=32, # Can be adjusted
        num_workers=2,  # Can be adjusted
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=test_ann_file_path, # Use the provided test annotation file
            metainfo=cfg.get('cephalometric_metainfo'), # Ensure metainfo is correctly referenced from main config
            pipeline=test_pipeline,
            test_mode=True, # Crucial for dataset to know it's in test mode
        )
    )

    # --- Override/Set up Test Evaluator for MRE (EPE) ---
    print("Setting up test evaluator with NME.")
    # Using just NME, which is the most appropriate for cephalometric landmarks
    cfg.test_evaluator = dict(
        type='NME',  # Normalized Mean Error
        norm_mode='keypoint_distance', # or 'bbox_size'
    )
    
    # --- Set up Test Configuration for the Runner ---
    cfg.test_cfg = dict() # Needs to be present

    # --- Set Checkpoint and Work Directory ---
    cfg.load_from = checkpoint_file_path
    cfg.work_dir = osp.join(work_dir, osp.splitext(osp.basename(config_file_path))[0])
    os.makedirs(cfg.work_dir, exist_ok=True)
    print(f"Test work directory set to: {cfg.work_dir}")

    # --- Build the Runner and Run Evaluation ---
    print("Building runner...")
    # Use our custom runner to handle checkpoint loading restrictions in PyTorch 2.6+
    cfg.runner_type = 'CustomRunner'
    runner = RUNNERS.build(cfg)
    
    print(f"Starting evaluation with checkpoint: {checkpoint_file_path}")
    # Load the checkpoint directly, avoiding the Runner.load_checkpoint method
    # This isn't ideal, but we're working around limitations in MMEngine with PyTorch 2.6+
    try:
        print("Attempting to load checkpoint...")
        runner.load_checkpoint(checkpoint_file_path)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("\nFallback: You can try manually loading this checkpoint in a separate script that adds")
        print("the necessary MMEngine classes to PyTorch's safe globals list:")
        print("```python")
        print("import torch.serialization")
        print("from mmengine.config import ConfigDict")
        print("torch.serialization.add_safe_globals([ConfigDict])")
        print("```")
        return None
        
    metrics = runner.test()

    print("\n--- Evaluation Results ---")
    if metrics:
        # Check for NME metric
        nme_value = metrics.get('NME', None)
        
        if nme_value is not None:
            print(f"Normalized Mean Error (NME): {nme_value:.4f}")
            # NME is similar to MRE but normalized by a reference distance
            print("This is equivalent to a Mean Radial Error normalized by inter-keypoint distance.")
        else:
            print("Available metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
        
        return metrics
    else:
        print("Evaluation did not return any metrics.")
        return None

if __name__ == '__main__':
    print("Starting evaluation script...")
    # Create dummy files and directories for example usage
    # In a real scenario, these paths would point to your actual files.

    # --- CONFIGURATION FOR EXAMPLE --- 
    # This assumes that hrnetv2_w18_cephalometric_224x224.py is in ./configs/hrnetv2/
    # and _base_ files are in ./configs/_base_/
    # Also assumes cephalometric_dataset_info.py, custom_*.py are in the current dir.
    
    EXAMPLE_CONFIG_PATH = 'configs/hrnetv2/hrnetv2_w18_cephalometric_224x224.py'
    # Replace with the actual path to your best checkpoint
    EXAMPLE_CHECKPOINT_PATH = "work_dirs/hrnetv2_w18_cephalometric_experiment/epoch_60.pth" # Update this to your best checkpoint
    EXAMPLE_TEST_ANN_FILE = "train_data_pure_old_numpy.json" # Use the main JSON file
    EXAMPLE_DATA_ROOT = "/content/drive/MyDrive/Lala's Masters/" # Your data root
    TEMP_EVAL_DIR = "temp_eval_dir_example"

    # --- Check if example files exist before running --- 
    if not osp.exists(EXAMPLE_CONFIG_PATH):
        print(f"ERROR: Example config file not found: {EXAMPLE_CONFIG_PATH}")
        print("Please ensure the config file path is correct and it imports _base_ configs correctly.")
        print("You might need to create a `configs/_base_` directory with `default_runtime.py`.")
        exit()
    
    if not osp.exists(EXAMPLE_CHECKPOINT_PATH):
        print(f"ERROR: Example checkpoint file not found: {EXAMPLE_CHECKPOINT_PATH}")
        print("Please update EXAMPLE_CHECKPOINT_PATH to your trained model.")
        exit()

    if not osp.exists(osp.join(EXAMPLE_DATA_ROOT, EXAMPLE_TEST_ANN_FILE)):
        print(f"ERROR: Example test annotation file not found at: {osp.join(EXAMPLE_DATA_ROOT, EXAMPLE_TEST_ANN_FILE)}")
        print("Please ensure your test JSON file exists and paths are correct.")
        exit()
    
    # Create a dummy configs/_base_ directory and default_runtime.py if they don't exist for the example to run
    base_config_dir = osp.join(osp.dirname(EXAMPLE_CONFIG_PATH), '../_base_') # Should resolve to ./configs/_base_
    os.makedirs(base_config_dir, exist_ok=True)
    default_runtime_path = osp.join(base_config_dir, 'default_runtime.py')
    if not osp.exists(default_runtime_path):
        print(f"Creating dummy {default_runtime_path} for example to run...")
        with open(default_runtime_path, 'w') as f:
            f.write("default_scope = 'mmpose'\n")
            f.write("default_hooks = dict(timer=dict(type='IterTimerHook'), logger=dict(type='LoggerHook', interval=50))\n")
            f.write("env_cfg = dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), dist_cfg=dict(backend='nccl'))\n")
            f.write("log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)\n")
            f.write("log_level = 'INFO'\n")
            f.write("load_from = None\n")
            f.write("resume = False\n")
            f.write("randomness = dict(seed=None, diff_rank_seed=False, deterministic=False)\n")

    print("\nRunning MRE evaluation function with example paths...")
    evaluate_checkpoint_mre(
        config_file_path=EXAMPLE_CONFIG_PATH,
        checkpoint_file_path=EXAMPLE_CHECKPOINT_PATH,
        test_ann_file_path=EXAMPLE_TEST_ANN_FILE,
        data_root=EXAMPLE_DATA_ROOT,
        work_dir=TEMP_EVAL_DIR
    )
    print("\nEvaluation script finished.") 