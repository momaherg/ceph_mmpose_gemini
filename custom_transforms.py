from mmpose.datasets.transforms.loading import LoadImage
from mmpose.registry import TRANSFORMS
import numpy as np
from mmengine.registry import init_default_scope

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

# Print confirmation that the module has been loaded and registration attempted
print(f"LoadImageNumpy registered: {TRANSFORMS.get('LoadImageNumpy') is not None}")

# This will output all registered transforms for debugging
print("Registered transforms:", list(TRANSFORMS.module_dict.keys()))

@TRANSFORMS.register_module(force=True)
class DebugInspectKeys(LoadImage): # Inherits from LoadImage just to have a base class
    def __init__(self, keys_to_inspect=None, shapes_only=True, **kwargs):
        super().__init__(**kwargs)
        self.keys_to_inspect = keys_to_inspect if keys_to_inspect else []
        self.shapes_only = shapes_only
        print(f"DebugInspectKeys initialized to inspect: {self.keys_to_inspect}")

    def transform(self, results: dict) -> dict:
        print("--- DebugInspectKeys: Inspecting results ---")
        for key in self.keys_to_inspect:
            if key in results:
                value = results[key]
                if self.shapes_only and hasattr(value, 'shape'):
                    print(f"  Key: '{key}', Shape: {value.shape}, Type: {type(value)}")
                else:
                    print(f"  Key: '{key}', Value: {value}, Type: {type(value)}")
            else:
                print(f"  Key: '{key}' not found in results.")
        print("--- DebugInspectKeys: Inspection done ---")
        return results 