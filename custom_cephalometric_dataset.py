import numpy as np
import pandas as pd
from mmengine.dataset import BaseDataset
from mmpose.registry import DATASETS
from cephalometric_dataset_info import dataset_info, landmark_names_in_order, original_landmark_cols

@DATASETS.register_module()
class CustomCephalometricDataset(BaseDataset):
    """Custom dataset for Cephalometric landmark detection.

    Args:
        ann_file (str): Annotation file path. Not used if data_df is provided.
        data_df (pd.DataFrame): Pandas DataFrame containing the dataset.
                                Required if ann_file is not to be used.
        # Other arguments from BaseDataset like pipeline, data_root, etc.
    """
    METAINFO: dict = dataset_info # Crucial for MMPose integration

    def __init__(self, 
                 ann_file='', 
                 data_df: pd.DataFrame = None, 
                 filter_cfg=None,
                 **kwargs):
        if data_df is None and not ann_file:
            raise ValueError("Either 'data_df' or 'ann_file' must be provided.")
        
        self.data_df = data_df
        # If ann_file is provided and data_df is None, you would typically load it here.
        # For this use case, we assume data_df is always passed directly from main.py

        super().__init__(ann_file=ann_file, filter_cfg=filter_cfg, **kwargs)

    def _load_data_list(self):
        """Load annotations from the provided DataFrame."""
        data_list = []
        if self.data_df is None:
            # This part would handle loading from ann_file if self.data_df was not provided
            # For now, it assumes self.data_df is available
            if self.ann_file:
                # Placeholder: logic to load self.data_df from self.ann_file if needed
                # For example, if ann_file is a path to a JSON that needs to be loaded into a DataFrame
                # self.data_df = pd.read_json(self.ann_file) # Or other appropriate reader
                print(f"Warning: data_df not provided directly, attempting to load from ann_file: {self.ann_file}")
                # This would typically be more robust, e.g. self.data_df = pd.read_json(self.data_root + self.ann_file)
                # For now, we will skip if data_df is not present and ann_file is not handled properly
                pass # Let it proceed, it will likely fail if data_df remains None
            else:
                return [] # No data to load

        # Ensure the DataFrame is available
        if self.data_df is None:
            print("Error: DataFrame not available for loading data.")
            return []

        # The `landmark_cols` from the problem description contains 38 coordinate columns.
        # We need to map them to 19 keypoints (x,y pairs).
        # `landmark_names_in_order` from cephalometric_dataset_info.py defines the order for our (19,2) array.
        # `original_landmark_cols` provides the x,y column names in sequence.

        num_keypoints = len(self.METAINFO['keypoint_info'])

        for index, row in self.data_df.iterrows():
            img_array_list = row['Image'] # This is a list of lists/tuples for pixels
            # Convert to numpy array and reshape. Assuming a (H, W, C) format internally for images.
            # Images are 224x224x3. The list is flat (50176, 3).
            try:
                img_np = np.array(img_array_list, dtype=np.uint8).reshape((224, 224, 3))
            except Exception as e:
                print(f"Error processing image for patient_id {row.get('patient_id', 'Unknown')}: {e}")
                print(f"Image data was: {img_array_list[:5]}...") # Print a snippet
                continue # Skip this problematic entry
            
            keypoints = np.zeros((num_keypoints, 2), dtype=np.float32)
            keypoints_visible = np.ones(num_keypoints, dtype=np.int32) * 2 # Assume all keypoints are visible and not occluded initially

            for i, kp_name in enumerate(landmark_names_in_order):
                # Find the corresponding x and y columns from original_landmark_cols
                # The order in original_landmark_cols is [sella_x, sella_y, nasion_x, nasion_y, ...]
                # So, for the i-th keypoint name, its x-coord is at original_landmark_cols[i*2]
                # and y-coord is at original_landmark_cols[i*2 + 1]
                x_col = original_landmark_cols[i*2]
                y_col = original_landmark_cols[i*2+1]

                if x_col in row and y_col in row:
                    keypoints[i, 0] = row[x_col]
                    keypoints[i, 1] = row[y_col]
                else:
                    # This case should ideally not happen if data is clean
                    # Mark as not visible if data is missing
                    keypoints[i, 0] = 0 
                    keypoints[i, 1] = 0
                    keypoints_visible[i] = 0 
                    # print(f"Warning: Missing keypoint data for {kp_name} (cols {x_col}, {y_col}) in patient_id {row.get('patient_id')}")

            # Bounding box: Since images are already 224x224 and contain the subject,
            # the bbox can be considered the full image.
            # Format: [x1, y1, x2, y2] or [x, y, w, h] depending on pipeline expectations.
            # MMPose top-down typically expects [x, y, w, h] where (x,y) is top-left.
            # Or, if pipelines use `bbox_cs`, they might convert. Let's use [x1,y1,x2,y2] for now.
            # TopdownAffine expects bbox in (x, y, w, h)
            bbox_xywh = np.array([0, 0, 224, 224], dtype=np.float32)

            data_info = {
                'img': img_np, # The actual image numpy array
                'img_path': str(row.get('patient_id', f'index_{index}')), # Create a pseudo img_path for identification
                'img_id': str(row.get('patient_id', index)),
                'bbox': bbox_xywh, # Full image as bbox [x,y,w,h]
                'keypoints': keypoints,
                'keypoints_visible': keypoints_visible,
                'id': str(row.get('patient_id', index)), # Sample ID
                # Basic metadata that might be needed by some transforms or for debugging
                'ori_shape': (224, 224),
                'img_shape': (224, 224), # Shape of the image provided
                # 'dataset_name': self.METAINFO['dataset_name'] # Will be added by PackPoseInputs if not here
                # 'category_id': 1, # If you have a single category (e.g. person/patient)
                # Other relevant info from the row can be added here if needed by pipeline
                'patient_text_id': row.get('patient', ''),
                'set': row.get('set', 'train'), # From the 'set' column
                'class': row.get('class', None) # Diagnostic class
            }
            data_list.append(data_info)
        
        if not data_list:
            print("Warning: No data loaded. Check DataFrame or loading logic.")

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