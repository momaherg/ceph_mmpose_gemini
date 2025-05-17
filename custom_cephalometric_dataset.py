import numpy as np
import pandas as pd
from mmengine.dataset import BaseDataset
from mmpose.registry import DATASETS
from cephalometric_dataset_info import dataset_info, landmark_names_in_order, original_landmark_cols

@DATASETS.register_module()
class CustomCephalometricDataset(BaseDataset):
    """Custom dataset for Cephalometric landmark detection.

    Args:
        ann_file (str): Annotation file path.
        pipeline (list): Processing pipeline.
        metainfo (dict, optional): Metadata for the dataset. Defaults to None.
        test_mode (bool, optional): Whether the dataset is in test mode. Defaults to False.
        # Other arguments from BaseDataset like data_root, test_mode etc.
    """
    METAINFO: dict = dataset_info

    def __init__(self, 
                 ann_file: str, 
                 pipeline: list,
                 metainfo: dict = None, # Will be passed from config
                 test_mode: bool = False, # Will be passed from config
                 **kwargs):
        
        # BaseDataset will handle ann_file, pipeline, and merging METAINFO with passed metainfo.
        # It also sets self.test_mode based on the argument.
        super().__init__(ann_file=ann_file, 
                         pipeline=pipeline, 
                         metainfo=metainfo,
                         test_mode=test_mode,
                         **kwargs)

    def _load_data_list(self) -> list:
        """Load annotations from self.ann_file (set by BaseDataset)."""
        data_list = []
        
        print(f"Attempting to load custom annotations from: {self.ann_file}")
        try:
            current_df = pd.read_json(self.ann_file)
            print(f"Successfully loaded DataFrame from {self.ann_file}. Shape: {current_df.shape}")
        except Exception as e:
            raise IOError(f"Error loading annotation JSON file {self.ann_file}: {e}")

        if current_df.empty:
            print(f"Warning: DataFrame loaded from {self.ann_file} is empty.")
            return []

        num_keypoints = len(self.METAINFO['keypoint_info'])

        for index, row in current_df.iterrows():
            img_array_list = row['Image'] 
            try:
                img_np = np.array(img_array_list, dtype=np.uint8).reshape((224, 224, 3))
            except Exception as e:
                print(f"Error processing image for patient_id {row.get('patient_id', 'Unknown')} at index {index}: {e}")
                continue 
            
            keypoints = np.zeros((num_keypoints, 2), dtype=np.float32)
            keypoints_visible = np.ones(num_keypoints, dtype=np.int32) * 2 # Assume visible and not occluded initially

            for i, kp_name in enumerate(landmark_names_in_order):
                x_col = original_landmark_cols[i*2]
                y_col = original_landmark_cols[i*2+1]

                if x_col in row and y_col in row and pd.notna(row[x_col]) and pd.notna(row[y_col]):
                    keypoints[i, 0] = row[x_col]
                    keypoints[i, 1] = row[y_col]
                else:
                    keypoints[i, 0] = 0 # Or some other placeholder like -1, depending on pipeline expectations
                    keypoints[i, 1] = 0
                    keypoints_visible[i] = 0 # Mark as not visible and not labeled
            
            # Bbox in [x1, y1, x2, y2] format. Assuming full image is the bbox.
            bbox = np.array([0, 0, 224, 224], dtype=np.float32)

            data_info = {
                'img': img_np, # The actual image numpy array
                'img_path': str(row.get('patient_id', f'index_{index}')), # For metadata/logging
                'img_id': str(row.get('patient_id', index)), # MMPose often uses img_id
                'bbox': bbox, # Expected by top-down pipelines: [x1, y1, x2, y2]
                'keypoints': keypoints,
                'keypoints_visible': keypoints_visible,
                'id': str(row.get('patient_id', index)), # Your custom ID field
                # 'ori_shape' and 'img_shape' will be added by transforms like LoadImageNumpy
                'patient_text_id': row.get('patient', ''),
                'set': row.get('set', 'train'),
                'class_label': row.get('class', None) # Renamed from 'class'
            }
            data_list.append(data_info)
        
        if not data_list and not current_df.empty:
            print("Warning: _load_data_list resulted in an empty list, but the DataFrame was not empty. Check parsing logic.")
        
        print(f"_load_data_list finished. Parsed {len(data_list)} items.")
        return data_list

    # No need to override load_data_list(self) - BaseDataset will call _load_data_list.
    # No need to override get_data_info(self, idx) - BaseDataset.get_data_info will use self.data_list.
    # __getitem__ is handled by BaseDataset, which calls `self.pipeline(self.get_data_info(idx))`

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