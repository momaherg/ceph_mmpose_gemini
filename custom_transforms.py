from mmpose.datasets.transforms.loading import LoadImage
from mmpose.datasets.transforms.formatting import PackPoseInputs
from mmpose.registry import TRANSFORMS
import numpy as np
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData

# Initialize scope to ensure registry is ready
init_default_scope('mmpose')

print("Registering LoadImageNumpy in the TRANSFORMS registry...")

@TRANSFORMS.register_module(force=True) # Force registration to override any existing
class LoadImageNumpy(LoadImage):
    """Load an image from results['img'] which is already a numpy array.
    The 'img' key is populated by CustomCephalometricDataset.
    """
    
    def __init__(self, **kwargs):
        print("LoadImageNumpy.__init__ called")
        super().__init__(**kwargs)

    def transform(self, results: dict) -> dict:
        """Load an image from results['img'] which is already a numpy array.
        """
        img = results['img']
        if not isinstance(img, np.ndarray):
            raise TypeError(f"Image should be a NumPy array, but got {type(img)}")
        if img is None: # Should not happen if dataset guarantees 'img'
            raise ValueError('Image is not loaded, results["img"] is None.')

        # Assuming image from dataset is HWC.
        # MMPose pipelines usually handle channel order (e.g. BGR) and ToTensor conversion later.
        
        results['img_shape'] = img.shape[:2] # (h, w)
        results['ori_shape'] = img.shape[:2] # (h, w)
        # 'img_path' should be set by the dataset if needed for metainfo,
        # otherwise, it can be a placeholder or the 'patient_id'.
        # results['img_path'] = results.get('img_path', str(results.get('img_id', '')))
        return results

print("Registering CustomPackPoseInputs in the TRANSFORMS registry...")

@TRANSFORMS.register_module(force=True)
class CustomPackPoseInputs(PackPoseInputs):
    """Custom PackPoseInputs that properly handles bbox_scores and gt_classification for cephalometric dataset."""
    
    def transform(self, results: dict) -> dict:
        """Transform function to pack pose inputs, including bbox_scores and gt_classification."""
        
        # Call parent transform first
        packed_results = super().transform(results)
        
        # Ensure bbox_scores and gt_classification are properly added to data_samples
        if 'data_samples' in packed_results:
            data_sample = packed_results['data_samples']
            
            # Check if gt_instances exists and bbox_scores is in the original results
            if hasattr(data_sample, 'gt_instances') and 'bbox_scores' in results:
                gt_instances = data_sample.gt_instances
                
                # Add bbox_scores to gt_instances
                if isinstance(results['bbox_scores'], np.ndarray):
                    import torch
                    gt_instances.bbox_scores = torch.from_numpy(results['bbox_scores']).float()
                else:
                    gt_instances.bbox_scores = results['bbox_scores']
            
            # Add gt_classification to data_sample metainfo if available
            if 'gt_classification' in results and results['gt_classification'] is not None:
                # Set as metainfo to avoid conflicts with MMEngine's field management
                data_sample.set_metainfo({'gt_classification': results['gt_classification']})
        
        return packed_results

# Print confirmation that the module has been loaded and registration attempted
print(f"LoadImageNumpy registered: {TRANSFORMS.get('LoadImageNumpy') is not None}")

# This will output all registered transforms for debugging
print("Registered transforms:", list(TRANSFORMS.module_dict.keys())) 