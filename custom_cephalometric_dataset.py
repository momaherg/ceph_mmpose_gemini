import numpy as np
import pandas as pd
from mmengine.dataset import BaseDataset
from mmpose.registry import DATASETS
from cephalometric_dataset_info import dataset_info, landmark_names_in_order, original_landmark_cols

@DATASETS.register_module()
class CustomCephalometricDataset(BaseDataset):
    """Custom dataset for Cephalometric landmark detection.

    Args:
        ann_file_arg (str): Annotation file path. Used if data_df is not provided.
        data_df (pd.DataFrame): Pandas DataFrame containing the dataset.
        pipeline (list): Processing pipeline.
        filter_cfg (dict, optional): Config for filtering data. Defaults to None.
        # Other arguments from BaseDataset like data_root, test_mode etc.
    """
    METAINFO: dict = dataset_info

    def __init__(self, 
                 ann_file_arg='', # Renamed to avoid confusion with self.ann_file set by super
                 data_df: pd.DataFrame = None, 
                 filter_cfg=None,
                 pipeline=(), # Explicitly accept pipeline, default to empty tuple
                 **kwargs):
        
        if data_df is None and not ann_file_arg:
            raise ValueError("Either 'data_df' or 'ann_file_arg' must be provided.")
        
        self.data_df = data_df
        
        # Determine what ann_file to pass to the superclass
        actual_ann_file_for_super = None
        if self.data_df is not None:
            # If DataFrame is provided, tell superclass there's no annotation file path to load from.
            # Its load_data_list will then call our _load_data_list, which uses self.data_df.
            actual_ann_file_for_super = None 
        elif ann_file_arg: # If df not given, but ann_file_arg is, pass it to super
            actual_ann_file_for_super = ann_file_arg
        # If data_df is None and ann_file_arg is also empty, the ValueError above handles it.

        # BaseDataset __init__ expects metainfo to be passed if it's overriding class METAINFO
        # or it will use self.METAINFO. We set METAINFO as a class var, so it's fine.
        # It also expects pipeline. kwargs will catch others like data_root, test_mode.
        super().__init__(ann_file=actual_ann_file_for_super, 
                         filter_cfg=filter_cfg, 
                         pipeline=pipeline, 
                         **kwargs)

    def _load_data_list(self):
        """Load annotations, primarily from self.data_df, or from self.ann_file if df is not pre-loaded."""
        data_list = []
        
        if self.data_df is None:
            # This block executes if data_df was not passed to __init__ directly.
            # self.ann_file is what was passed to super().__init__.
            if self.ann_file and isinstance(self.ann_file, str):
                print(f"Initial self.data_df is None. Attempting to load DataFrame from self.ann_file: '{self.ann_file}'")
                try:
                    # Assuming self.ann_file is a complete path to a JSON file that pandas can read.
                    # If self.data_root is set (e.g. via kwargs from a config), and self.ann_file is relative,
                    # one might need to construct the full path: os.path.join(self.data_root, self.ann_file)
                    # For now, assume self.ann_file is directly usable by pd.read_json.
                    self.data_df = pd.read_json(self.ann_file)
                    print(f"Successfully loaded DataFrame from self.ann_file. Shape: {self.data_df.shape}")
                except Exception as e:
                    print(f"Error loading DataFrame from self.ann_file ('{self.ann_file}'): {e}")
                    # self.data_df remains None, and the check below will handle it.
            
            if self.data_df is None: # If still None (ann_file was None, empty, or loading failed)
                print("Error: DataFrame (self.data_df) is None and could not be loaded from self.ann_file.")
                return [] # Return empty list, subsequent processing will show 0 samples.

        # Proceed to process self.data_df (which is now guaranteed to be populated if we reach here)
        num_keypoints = len(self.METAINFO['keypoint_info'])

        for index, row in self.data_df.iterrows():
            img_array_list = row['Image'] 
            try:
                img_np = np.array(img_array_list, dtype=np.uint8).reshape((224, 224, 3))
            except Exception as e:
                print(f"Error processing image for patient_id {row.get('patient_id', 'Unknown')}: {e}")
                continue 
            
            keypoints = np.zeros((num_keypoints, 2), dtype=np.float32)
            keypoints_visible = np.ones(num_keypoints, dtype=np.int32) * 2 

            for i, kp_name in enumerate(landmark_names_in_order):
                x_col = original_landmark_cols[i*2]
                y_col = original_landmark_cols[i*2+1]

                if x_col in row and y_col in row and pd.notna(row[x_col]) and pd.notna(row[y_col]):
                    keypoints[i, 0] = row[x_col]
                    keypoints[i, 1] = row[y_col]
                else:
                    keypoints[i, 0] = 0 
                    keypoints[i, 1] = 0
                    keypoints_visible[i] = 0 
            
            bbox_xywh = np.array([0, 0, 224, 224], dtype=np.float32)

            data_info = {
                'img': img_np,
                'img_path': str(row.get('patient_id', f'index_{index}')),
                'img_id': str(row.get('patient_id', index)),
                'bbox': bbox_xywh,
                'keypoints': keypoints,
                'keypoints_visible': keypoints_visible,
                'id': str(row.get('patient_id', index)),
                'ori_shape': (224, 224),
                'img_shape': (224, 224),
                'patient_text_id': row.get('patient', ''),
                'set': row.get('set', 'train'),
                'class': row.get('class', None)
            }
            data_list.append(data_info)
        
        if not data_list and not self.data_df.empty:
            print("Warning: Data list is empty but DataFrame was not. Check processing logic in _load_data_list.")
        elif self.data_df.empty:
            print("Warning: DataFrame was empty, so data list is empty.")

        return data_list

    def get_data_info(self, idx: int) -> dict:
        """Get data information by index. Used by the BaseDataset for __getitem__.
           This needs to return a dictionary that includes the 'img' numpy array directly.
        """
        # `self.data_list` is populated by `_load_data_list` which is called by BaseDataset constructor
        data_info = super().get_data_info(idx)
        # The `img` key is already populated with the numpy array by our `_load_data_list`
        return data_info

    # __getitem__ is handled by BaseDataset, which calls `self.pipeline(self.get_data_info(idx))`
    # No need to override it unless very specific logic is needed for how pipeline is called. 