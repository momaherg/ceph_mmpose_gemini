import numpy as np
import pandas as pd
from mmengine.dataset import BaseDataset
from mmpose.registry import DATASETS
from cephalometric_dataset_info import dataset_info, landmark_names_in_order, original_landmark_cols

@DATASETS.register_module()
class CustomCephalometricDataset(BaseDataset):
    """Custom dataset for Cephalometric landmark detection.

    Args:
        ann_file (str, optional): Annotation file path. Defaults to ''.
            Used if data_df is not provided.
        data_df (pd.DataFrame, optional): Pandas DataFrame containing the dataset.
            Defaults to None.
        pipeline (list): Processing pipeline.
        filter_cfg (dict, optional): Config for filtering data. Defaults to None.
        **kwargs: Other arguments passed to BaseDataset.
    """
    METAINFO: dict = dataset_info

    def __init__(self, 
                 ann_file='', 
                 data_df: pd.DataFrame = None, 
                 pipeline=(), 
                 filter_cfg=None,
                 **kwargs):
        
        if data_df is None and not ann_file:
            # If using ann_file, it should be a valid path string. 
            # If it's an empty string and no data_df, BaseDataset might try to load from '' which is an error.
            # However, if ann_file is empty string, BaseDataset usually calls _load_data_list.
            # The critical part is to have a way for _load_data_list to then load the df if needed.
            pass # Allow BaseDataset to initialize with ann_file='', it will call load_data_list

        self.data_df = data_df

        # Pass pipeline and other relevant args to BaseDataset.
        # BaseDataset will set self.ann_file. If 'ann_file' is empty, it will try to call
        # self._load_data_list via its own self.load_data_list.
        super().__init__(ann_file=ann_file, # Pass the original ann_file string here
                         pipeline=pipeline, 
                         filter_cfg=filter_cfg, 
                         **kwargs)

    def load_data_list(self) -> list:
        """Load annotations.
        This method is responsible for returning the list of data items.
        It will use self._load_data_list() which can handle either a pre-loaded
        DataFrame or load from self.ann_file (resolved by BaseDataset).
        """
        if self.data_df is not None:
            print("CustomCephalometricDataset.load_data_list(): DataFrame was pre-loaded.")
        elif self.ann_file: # ann_file is a property of BaseDataset and should be the full path
            print(f"CustomCephalometricDataset.load_data_list(): DataFrame not pre-loaded, self.ann_file is '{self.ann_file}'. Will attempt to load.")
        else:
            print("CustomCephalometricDataset.load_data_list(): DataFrame not pre-loaded and no self.ann_file path seems available for _load_data_list. This might cause issues in _load_data_list.")
        
        data_list = self._load_data_list()
        
        if not data_list and (self.data_df is not None and not self.data_df.empty):
             print("Warning: CustomCephalometricDataset.load_data_list() is returning an empty list, but a DataFrame was available and not empty.")
        elif not data_list and not self.ann_file:
             print("Warning: CustomCephalometricDataset.load_data_list() is returning an empty list, and no ann_file_path was set for loading.")
        elif not data_list:
             print("Warning: CustomCephalometricDataset.load_data_list() is returning an empty list.")

        return data_list

    def _load_data_list(self) -> list:
        """Load annotations from self.data_df or from self.ann_file if df not pre-loaded."""
        data_list = []
        
        current_df = self.data_df

        if current_df is None:
            if self.ann_file: # Use self.ann_file (should be full path from BaseDataset)
                print(f"Initial self.data_df is None. Attempting to load DataFrame from self.ann_file: '{self.ann_file}'")
                try:
                    current_df = pd.read_json(self.ann_file)
                    print(f"Successfully loaded DataFrame from self.ann_file. Shape: {current_df.shape}")
                except FileNotFoundError:
                    print(f"FileNotFoundError: Cannot find annotation file at '{self.ann_file}'. Please check data_root and ann_file in your config.")
                    return []
                except Exception as e:
                    print(f"Error loading DataFrame from self.ann_file ('{self.ann_file}'): {e}")
                    return [] 
            else:
                print("Error: DataFrame is None and self.ann_file is not set.")
                return []
        
        if current_df is None: # Should not happen if logic above is correct
             print("Critical Error: current_df is still None in _load_data_list.")
             return []

        num_keypoints = len(self.METAINFO['keypoint_info'])

        for index, row in current_df.iterrows():
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
            
            # Ensure bbox has the right shape for a single instance: (1, 4) instead of (4,)
            bbox = np.array([[0, 0, 224, 224]], dtype=np.float32)  # Note the extra brackets to make it (1, 4)
            
            # Add bbox_scores (confidence scores for bounding boxes)
            bbox_scores = np.array([1.0], dtype=np.float32)  # High confidence since we use the full image

            data_info = {
                'img': img_np,
                'img_path': str(row.get('patient_id', f'index_{index}')),
                'img_id': str(row.get('patient_id', index)),
                'bbox': bbox,
                'bbox_scores': bbox_scores,  # Add bbox scores for validation
                'keypoints': keypoints.reshape(1, num_keypoints, 2),  # Reshape to (1, K, 2) for single instance
                'keypoints_visible': keypoints_visible.reshape(1, num_keypoints),  # Reshape to (1, K) for single instance
                'id': str(row.get('patient_id', index)),
                'ori_shape': (224, 224),
                'img_shape': (224, 224),
                'patient_text_id': row.get('patient', ''),
                'set': row.get('set', 'train'),
                'class': row.get('class', None)
            }
            data_list.append(data_info)
        
        if not data_list and not current_df.empty:
            print("Warning: Data list is empty but DataFrame was not. Check processing logic in _load_data_list.")
        elif current_df.empty:
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