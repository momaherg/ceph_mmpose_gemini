# !pip install psutil ninja

# !pip3 install openmim
# !mim install numpy==1.26.4
# !mim install mmengine
# !mim install "mmcv==2.1.0"
# !mim install "mmpose>=1.1.0"
# !mim install mmdet==3.2.0

import os
import json
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image as PILImage
import subprocess # Added for launching training

# --- Configuration Constants ---
LANDMARK_COLS = ['sella_x', 'sella_y', 'nasion_x', 'nasion_y', 'A point_x', 'A point_y',
                 'B point_x', 'B point_y', 'upper 1 tip_x', 'upper 1 tip_y',
                 'upper 1 apex_x', 'upper 1 apex_y', 'lower 1 tip_x', 'lower 1 tip_y',
                 'lower 1 apex_x', 'lower 1 apex_y', 'ANS_x', 'ANS_y', 'PNS_x', 'PNS_y',
                 'Gonion_x', 'Gonion_y', 'Menton_x', 'Menton_y', 'ST Nasion_x',
                 'ST Nasion_y', 'Tip of the nose_x', 'Tip of the nose_y', 'Subnasal_x',
                 'Subnasal_y', 'Upper lip_x', 'Upper lip_y', 'Lower lip_x',
                 'Lower lip_y', 'ST Pogonion_x', 'ST Pogonion_y', 'gnathion_x',
                 'gnathion_y']
NUM_KEYPOINTS = len(LANDMARK_COLS) // 2
IMG_HEIGHT, IMG_WIDTH = 224, 224

DATA_ROOT = 'mmpose_ceph_data'
IMAGES_TRAIN_DIR = os.path.join(DATA_ROOT, 'images_train')
IMAGES_VAL_DIR = os.path.join(DATA_ROOT, 'images_val')
ANNOTATIONS_DIR = os.path.join(DATA_ROOT, 'annotations')

# --- Helper Functions ---

def create_dummy_data(num_samples=100):
    data = []
    keypoint_names_flat = LANDMARK_COLS
    for i in range(num_samples):
        img_array = np.random.randint(0, 256, size=(IMG_HEIGHT * IMG_WIDTH, 3), dtype=np.uint8)
        record = {
            'patient_id': i,
            'Image': img_array,
            'set': 'train' if i < num_samples * 0.8 else ('dev' if i < num_samples * 0.9 else 'test'),
            'class': np.random.choice([1, 2, 3])
        }
        for k_idx in range(NUM_KEYPOINTS):
            record[keypoint_names_flat[k_idx*2]] = np.random.uniform(10, IMG_WIDTH - 10)
            record[keypoint_names_flat[k_idx*2+1]] = np.random.uniform(10, IMG_HEIGHT - 10)
        data.append(record)
    return pd.DataFrame(data)

def process_dataframe_for_mmpose(df, landmark_cols_list, img_dir, set_name):
    mmpose_annotations = []
    print(f"Processing {set_name} data ({len(df)} samples)...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_array_flat = np.array(row['Image'])
        if img_array_flat.shape != (IMG_HEIGHT * IMG_WIDTH, 3):
            print(f"Warning: Image for patient {row['patient_id']} has unexpected flat shape {img_array_flat.shape}. Skipping.")
            continue
        try:
            img_array_hwc = img_array_flat.reshape(IMG_HEIGHT, IMG_WIDTH, 3)
        except ValueError as e:
            print(f"Error reshaping image for patient {row['patient_id']}: {e}. Flat shape {img_array_flat.shape}. Skipping.")
            continue

        if img_array_hwc.dtype != np.uint8:
            img_array_hwc = img_array_hwc.astype(np.uint8)

        img_pil = PILImage.fromarray(img_array_hwc)
        img_filename = f"patient_{row['patient_id']}.png"
        img_path = os.path.join(img_dir, img_filename)
        img_pil.save(img_path)

        keypoints = []
        keypoints_visible = []
        for i in range(NUM_KEYPOINTS):
            x_col, y_col = landmark_cols_list[i*2], landmark_cols_list[i*2+1]
            if x_col not in row or y_col not in row or pd.isna(row[x_col]) or pd.isna(row[y_col]):
                keypoints.extend([0.0, 0.0])
                keypoints_visible.append(0)
                continue
            x, y = float(row[x_col]), float(row[y_col])
            keypoints.extend([x, y])
            keypoints_visible.append(2)

        kps_np = np.array(keypoints).reshape(-1, 2)
        vis_np = np.array(keypoints_visible)
        visible_kps = kps_np[vis_np > 0]

        if len(visible_kps) > 0:
            x_min, y_min = visible_kps.min(axis=0)
            x_max, y_max = visible_kps.max(axis=0)
            padding = 10
            bbox_xywh = [
                max(0, x_min - padding),
                max(0, y_min - padding),
                min(IMG_WIDTH - (x_min - padding), (x_max - x_min) + 2 * padding),
                min(IMG_HEIGHT - (y_min - padding), (y_max - y_min) + 2 * padding)
            ]
        else:
            bbox_xywh = [0.0, 0.0, float(IMG_WIDTH), float(IMG_HEIGHT)]

        ann_entry = {
            'img_path': os.path.relpath(img_path, DATA_ROOT),
            'img_id': int(row['patient_id']),
            'bbox': bbox_xywh,
            'keypoints': np.array(keypoints).reshape(-1, 2).tolist(),
            'keypoints_visible': keypoints_visible,
            'id': idx, # Unique ID for the annotation instance (can be df index or a new counter)
        }
        mmpose_annotations.append(ann_entry)
    return mmpose_annotations

# --- Main Script Functions ---

def prepare_cephalometric_data():
    """Loads data, processes it, and saves images and annotations for MMPose."""
    print("Step 1: Preparing Cephalometric Data for MMPose...")

    # Load original data (replace with your actual loading if not using the global)
    try:
        # Attempt to use a globally defined train_data_pure if available
        # This is for compatibility if the DataFrame is loaded outside this function
        # in some environments (like notebooks).
        global train_data_pure 
        if 'train_data_pure' not in globals() or not isinstance(train_data_pure, pd.DataFrame):
            print("Global `train_data_pure` not found or not a DataFrame. Loading from specified JSON path...")
            train_data_pure = pd.read_json("/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json")
    except FileNotFoundError:
        print("Specified JSON file not found. Using dummy data.")
        train_data_pure = create_dummy_data(num_samples=150)
    except Exception as e:
        print(f"Error loading data: {e}. Using dummy data.")
        train_data_pure = create_dummy_data(num_samples=150)
        
    print(f"Using data with {len(train_data_pure)} samples.")

    keypoint_names = [col.replace('_x', '').replace('_y', '') for col in LANDMARK_COLS[::2]]
    print(f"Number of keypoints: {NUM_KEYPOINTS}")
    print(f"Keypoint names: {keypoint_names}")

    if os.path.exists(DATA_ROOT):
        print(f"Cleaning up existing data directory: {DATA_ROOT}")
        shutil.rmtree(DATA_ROOT)
    os.makedirs(IMAGES_TRAIN_DIR, exist_ok=True)
    os.makedirs(IMAGES_VAL_DIR, exist_ok=True)
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

    if 'set' not in train_data_pure.columns:
        print("Column 'set' not found. Creating a random 80/20 train/val split.")
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(train_data_pure, test_size=0.2, random_state=42)
    else:
        train_df = train_data_pure[train_data_pure['set'] == 'train'].copy()
        val_df = train_data_pure[train_data_pure['set'] == 'dev'].copy()
        if len(val_df) == 0:
            print("No 'dev' set found. Splitting 'train' set for validation (80/20).")
            if len(train_df) < 10:
                val_df = train_df.copy()
            else:
                from sklearn.model_selection import train_test_split
                train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
    print(f"Number of training samples: {len(train_df)}")
    print(f"Number of validation samples: {len(val_df)}")

    train_ann_data = process_dataframe_for_mmpose(train_df, LANDMARK_COLS, IMAGES_TRAIN_DIR, "train")
    val_ann_data = process_dataframe_for_mmpose(val_df, LANDMARK_COLS, IMAGES_VAL_DIR, "validation")

    with open(os.path.join(ANNOTATIONS_DIR, 'train_annotations.json'), 'w') as f:
        json.dump(train_ann_data, f)
    with open(os.path.join(ANNOTATIONS_DIR, 'val_annotations.json'), 'w') as f:
        json.dump(val_ann_data, f)
    print("Data preparation complete.")

def launch_mmpose_training():
    """Launches MMPose training using the configured setup."""
    print("Step 2: Launching MMPose Training...")

    config_file = 'configs/cephalometric/hrnetv2_w18_cephalometric_256x256.py'
    work_dir = 'work_dirs/hrnetv2_w18_cephalometric_256x256'

    # Adjust PYTHONPATH for the subprocess to find custom modules like cephalometric_dataset.py
    env = os.environ.copy()
    current_dir = os.path.abspath(os.path.dirname(__file__)) # or os.getcwd() if script is at root
    
    # Ensure the directory containing 'cephalometric_dataset.py' is in PYTHONPATH
    # Assuming cephalometric_dataset.py is in the same directory as this script, or workspace root.
    # If main.py is in workspace root, '.' is fine.
    # If main.py is in a subdir, adjust `module_dir_path` accordingly or use absolute path.
    module_dir_path = current_dir # Assuming cephalometric_dataset.py is here or found via .
                                  # This should be the workspace root.

    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{module_dir_path}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = module_dir_path
    
    print(f"PYTHONPATH for subprocess: {env.get('PYTHONPATH')}")
    os.makedirs(work_dir, exist_ok=True)

    # Determine the command to run for training
    cmd = None
    mmpose_train_executable = shutil.which('mmpose_train')
    mim_executable = shutil.which('mim')

    if mmpose_train_executable:
        print(f"Found `mmpose_train` executable at: {mmpose_train_executable}")
        cmd = [
            mmpose_train_executable,
            config_file,
            '--work-dir', work_dir,
            '--launcher', 'none' # For single GPU training
        ]
    elif mim_executable:
        print(f"Found `mim` executable at: {mim_executable}. Will use 'mim train mmpose ...'")
        cmd = [
            mim_executable, 'train', 'mmpose',
            config_file,
            '--work-dir', work_dir,
            '--launcher', 'none' # For single GPU training
        ]
    else:
        print("Neither `mmpose_train` nor `mim` found in PATH. Falling back to direct script execution.")
        # Fallback: Try to find local train.py script (original logic)
        mmpose_train_script_candidate1 = os.path.join(current_dir, 'mmpose', 'tools', 'train.py')
        mmpose_train_script_candidate2 = os.path.join(current_dir, 'tools', 'train.py')
        
        mmpose_train_script_path = None
        if os.path.exists(mmpose_train_script_candidate1):
            mmpose_train_script_path = mmpose_train_script_candidate1
        elif os.path.exists(mmpose_train_script_candidate2):
            mmpose_train_script_path = mmpose_train_script_candidate2
        
        if not mmpose_train_script_path:
            print(f"Warning: MMPose train script also not found at expected local paths:")
            print(f"  - {mmpose_train_script_candidate1}")
            print(f"  - {mmpose_train_script_candidate2}")
            print("Please ensure MMPose is cloned (e.g., as 'mmpose' in the project root) or adjust the path.")
            print("Attempting to use a generic 'tools/train.py' which will likely fail.")
            mmpose_train_script_path = 'tools/train.py' # Last resort fallback
        else:
            print(f"Found local MMPose train script at: {mmpose_train_script_path}")

        cmd = [
            'python', # Or 'python3', or sys.executable
            mmpose_train_script_path,
            config_file,
            '--work-dir', work_dir,
            '--launcher', 'none' # For single GPU training
        ]

    print(f"Executing training command: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        for line in process.stdout:
            print(line, end='')
        process.wait()
        if process.returncode != 0:
            print(f"Training process failed with return code {process.returncode}")
        else:
            print("Training process completed successfully.")
    except FileNotFoundError:
        # This specific error might be less likely now if cmd[0] is found by shutil.which
        # but could occur if the fallback tools/train.py is used and not found.
        executable_name = cmd[0] if cmd and cmd[0] else "[command not set]"
        print(f"Error: Failed to find the training script or executable '{executable_name}'.")
        print("Ensure MMPose is installed correctly and accessible in your PATH or cloned locally.")
    except Exception as e:
        print(f"An error occurred during the training subprocess: {e}")

# --- Script Entry Point ---
if __name__ == "__main__":
    # This global variable is used if the script is run in an environment (e.g. notebook)
    # where train_data_pure might be pre-loaded.
    # If not pre-loaded, prepare_cephalometric_data will try to load it from JSON or use dummy.
    train_data_pure = None # Initialize to None

    # Step 1: Prepare data
    prepare_cephalometric_data()

    # Step 2: Launch Training
    launch_mmpose_training()

    print("\nAll steps triggered.")
