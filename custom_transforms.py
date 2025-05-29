from mmpose.datasets.transforms.loading import LoadImage
from mmpose.datasets.transforms.formatting import PackPoseInputs
from mmpose.registry import TRANSFORMS
import numpy as np
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData
from mmcv.transforms import BaseTransform

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
    """Custom PackPoseInputs that properly handles bbox_scores for cephalometric dataset."""
    
    def transform(self, results: dict) -> dict:
        """Transform function to pack pose inputs, including bbox_scores."""
        
        # Call parent transform first
        packed_results = super().transform(results)
        
        # Ensure bbox_scores are properly added to gt_instances
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
                
                # Debug print removed - bbox_scores successfully added
        
        return packed_results

# Print confirmation that the module has been loaded and registration attempted
print(f"LoadImageNumpy registered: {TRANSFORMS.get('LoadImageNumpy') is not None}")

# This will output all registered transforms for debugging
print("Registered transforms:", list(TRANSFORMS.module_dict.keys()))

@TRANSFORMS.register_module()
class AddInputMeta(BaseTransform):
    """A transform to explicitly add 'input_center', 'input_scale', 
    and 'input_size' to the results dictionary.
    These keys are often required by downstream processes, like storing
    predictions relative to the transformed input space.
    """

    def __init__(self,
                 input_size_from_config: tuple,
                 use_shape_from_img_key: bool = False,
                 img_key: str = 'img'):
        """
        Args:
            input_size_from_config (tuple): The (W, H) input size used by
                transforms like TopdownAffine. This will be stored as
                metainfo['input_size'].
            use_shape_from_img_key (bool): If True, derives 'input_size' from
                the shape of results[img_key] instead of input_size_from_config.
                This can be useful if the actual tensor shape after transforms
                is preferred. Defaults to False.
            img_key (str): The key for the image tensor in results, used if
                use_shape_from_img_key is True. Defaults to 'img'.
        """
        super().__init__()
        self.input_size_from_config = tuple(input_size_from_config)
        self.use_shape_from_img_key = use_shape_from_img_key
        self.img_key = img_key

    def transform(self, results: dict) -> dict:
        """
        Args:
            results (dict): The results dictionary.
                Expected to contain 'center' and 'scale' (e.g., from
                GetBBoxCenterScale).

        Returns:
            dict: The results dictionary updated with 'input_center',
                  'input_scale', and 'input_size'.
        """
        if 'center' in results:
            results['input_center'] = np.array(
                results['center'], dtype=np.float32)
        else:
            # Fallback or warning if 'center' is missing, though it's expected
            # For example, if bbox is [0,0,W,H], center is [W/2, H/2]
            # This depends on how 'center' is normally derived if not from GetBBoxCenterScale
            # For now, we assume GetBBoxCenterScale provides it.
            pass

        if 'scale' in results:
            results['input_scale'] = np.array(
                results['scale'], dtype=np.float32)
        else:
            # Fallback or warning if 'scale' is missing
            pass

        if self.use_shape_from_img_key and self.img_key in results:
            # Image shape is typically (H, W, C) or (H, W)
            img_tensor = results[self.img_key]
            # MMPose input_size is (W, H)
            results['input_size'] = tuple(img_tensor.shape[1::-1])
        else:
            results['input_size'] = self.input_size_from_config
            
        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"input_size_from_config={self.input_size_from_config}, "
                f"use_shape_from_img_key={self.use_shape_from_img_key}, "
                f"img_key='{self.img_key}')") 