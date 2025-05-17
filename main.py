import pandas as pd
import numpy as np
import mmengine
from mmengine.registry import init_default_scope

# Import the custom dataset class and dataset_info (though dataset_info is mainly used by the class itself)
from custom_cephalometric_dataset import CustomCephalometricDataset
from cephalometric_dataset_info import dataset_info # For reference or direct use if needed

# Initialize the default scope for MMEngine/MMPose registries if not already done
# This is important for the @DATASETS.register_module() to work correctly if you
# were to load this dataset via a config file later.
# For direct instantiation like here, it's less critical but good practice.
init_default_scope('mmpose') # Or a custom scope if you have one

def main():
    print("Loading training data...")
    # Load the training data from the JSON file
    # Make sure this path is correct for your Jupyter environment
    json_file_path = "/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json"
    try:
        train_df = pd.read_json(json_file_path)
        print(f"Successfully loaded data from {json_file_path}. Shape: {train_df.shape}")
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        print("Please ensure the file path is correct and the file is accessible.")
        print("If running locally and not in Colab/Drive, adjust the path accordingly.")
        return

    print("Initializing CustomCephalometricDataset...")

    # Define a minimal pipeline for demonstration purposes.
    # In a real scenario, this would come from a config file.
    # This pipeline assumes 'img' is a NumPy array, which our custom dataset provides.
    pipeline = [
        dict(type='LoadImage'), # This is a new transform to handle the 'img' field as numpy array
        dict(type='KeypointConverter', num_keypoints=19, 
             # From (x,y) to (x,y,visibility=1) if not already done by dataset
             # Our dataset already provides keypoints_visible, so this might just reformat
             # or could be more specific depending on what PackPoseInputs expects.
             # Let's simplify and assume PackPoseInputs handles it correctly based on keypoints and keypoints_visible.
            ),
        dict(type='PackPoseInputs', 
             meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                        'bbox', 'id', 'patient_text_id', 'set', 'class')
            ) # Packs data into the format required by the model
    ]
    
    # We need a dummy LoadImage transform that handles numpy arrays if the default one expects file paths
    # Let's define one or use an existing one if available that just passes through numpy arrays.
    # For now, we will create a simple one.
    from mmpose.datasets.transforms.loading import LoadImage
    from mmpose.registry import TRANSFORMS

    @TRANSFORMS.register_module(name='LoadImageNumpy') # Register with a unique name
    class LoadImageNumpy(LoadImage):
        def transform(self, results: dict) -> dict:
            """Load an image from results['img'] which is already a numpy array."""
            # The 'img' key is already populated with a NumPy array by our CustomCephalometricDataset
            # This transform might just ensure it's in the expected format (e.g., BGR)
            # or do nothing if the format is already correct.
            img = results['img']
            if img is None:
                raise ValueError('Image is not loaded.')
            
            # Assuming image from dataset is HWC, RGB. MMPose typically expects BGR.
            # img = img[..., ::-1] # Convert RGB to BGR if needed
            # For now, let's assume the custom dataset provides it in the desired channel order
            # or that subsequent transforms handle it if necessary (e.g. ` kleuren.ToTensor` might infer)

            results['img_shape'] = img.shape[:2]
            results['ori_shape'] = img.shape[:2]
            # 'img_path' is already set by the dataset
            return results

    # Update pipeline to use our numpy loader
    pipeline[0] = dict(type='LoadImageNumpy')

    # Instantiate the custom dataset
    # No ann_file is needed as we pass data_df directly.
    # data_root can be omitted if not used by the dataset for resolving paths.
    custom_dataset = CustomCephalometricDataset(
        data_df=train_df,
        pipeline=pipeline,
        # metainfo_file is not directly a param of BaseDataset constructor, 
        # METAINFO is a class attribute. But BaseDataset uses self.METAINFO.
        # test_mode=False # Default
    )

    print(f"Custom dataset initialized. Number of samples: {len(custom_dataset)}")

    if len(custom_dataset) > 0:
        print("\nFetching a sample from the dataset...")
        try:
            sample = custom_dataset[0] # Get the first sample
            print("Sample fetched successfully!")
            print("Keys in sample:", sample.keys())
            print("Input tensor shape ('inputs'):", sample['inputs'].shape if isinstance(sample['inputs'], np.ndarray) or hasattr(sample['inputs'], 'shape') else type(sample['inputs']))
            print("Data Sample keys ('data_samples'):", sample['data_samples'].__dict__.keys() if hasattr(sample['data_samples'], '__dict__') else "N/A")
            
            # Further inspection of data_sample contents
            data_sample_content = sample['data_samples']
            print("  `gt_instances` in data_sample:", hasattr(data_sample_content, 'gt_instances'))
            if hasattr(data_sample_content, 'gt_instances'):
                print("    `keypoints` shape in gt_instances:", data_sample_content.gt_instances.keypoints.shape)
                print("    `keypoints_visible` shape in gt_instances:", data_sample_content.gt_instances.keypoints_visible.shape)
                print("    `bboxes` in gt_instances:", data_sample_content.gt_instances.bboxes)

            print("  `metainfo` in data_sample:", data_sample_content.metainfo)

        except Exception as e:
            print(f"Error fetching or inspecting sample: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Dataset is empty, cannot fetch a sample.")

if __name__ == '__main__':
    main()
