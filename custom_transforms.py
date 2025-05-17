from mmpose.datasets.transforms.loading import LoadImage
from mmpose.registry import TRANSFORMS
import numpy as np

@TRANSFORMS.register_module() # Default name will be 'LoadImageNumpy'
class LoadImageNumpy(LoadImage):
    def transform(self, results: dict) -> dict:
        """Load an image from results['img'] which is already a numpy array.
        The 'img' key is populated by CustomCephalometricDataset.
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