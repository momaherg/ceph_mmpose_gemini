from mmpose.datasets.transforms.loading import LoadImage
from mmpose.datasets.transforms.formatting import PackPoseInputs
from mmpose.registry import TRANSFORMS
import numpy as np
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData
import cv2
from scipy.ndimage import gaussian_filter
import random

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

print("Registering PhotoMetricDistortion in the TRANSFORMS registry...")

@TRANSFORMS.register_module(force=True)
class PhotoMetricDistortion:
    """Photometric distortion transform for data augmentation.
    
    Args:
        brightness_delta (int): Delta for brightness adjustment.
        contrast_range (tuple): Range for contrast adjustment.
        saturation_range (tuple): Range for saturation adjustment.  
        hue_delta (int): Delta for hue adjustment.
    """
    
    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_delta = hue_delta
    
    def __call__(self, results: dict) -> dict:
        """Apply photometric distortion to image."""
        return self.transform(results)
    
    def transform(self, results: dict) -> dict:
        """Apply photometric distortion to image."""
        img = results['img']
        
        # Convert to float for processing
        img = img.astype(np.float32)
        
        # Random brightness adjustment
        if random.random() < 0.5:
            delta = random.uniform(-self.brightness_delta, self.brightness_delta)
            img += delta
        
        # Random contrast adjustment
        if random.random() < 0.5:
            alpha = random.uniform(self.contrast_range[0], self.contrast_range[1])
            img *= alpha
        
        # Convert to HSV for saturation and hue adjustments
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Ensure values are in [0, 255] range before HSV conversion
            img = np.clip(img, 0, 255)
            hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Random saturation adjustment
            if random.random() < 0.5:
                saturation_factor = random.uniform(self.saturation_range[0], self.saturation_range[1])
                hsv[:, :, 1] *= saturation_factor
            
            # Random hue adjustment
            if random.random() < 0.5:
                hue_delta = random.uniform(-self.hue_delta, self.hue_delta)
                hsv[:, :, 0] = (hsv[:, :, 0] + hue_delta) % 180
            
            # Convert back to RGB
            hsv = np.clip(hsv, 0, [179, 255, 255])
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        
        # Clip final values and convert back to uint8
        img = np.clip(img, 0, 255).astype(np.uint8)
        results['img'] = img
        
        return results

print("Registering RandomGaussianNoise in the TRANSFORMS registry...")

@TRANSFORMS.register_module(force=True)
class RandomGaussianNoise:
    """Add random Gaussian noise to images.
    
    Args:
        std_range (tuple): Range for noise standard deviation.
        prob (float): Probability of applying noise.
    """
    
    def __init__(self, std_range=(0.0, 0.1), prob=0.3):
        self.std_range = std_range
        self.prob = prob
    
    def __call__(self, results: dict) -> dict:
        """Apply Gaussian noise to image."""
        return self.transform(results)
    
    def transform(self, results: dict) -> dict:
        """Apply Gaussian noise to image."""
        if random.random() < self.prob:
            img = results['img'].astype(np.float32)
            std = random.uniform(self.std_range[0], self.std_range[1])
            noise = np.random.normal(0, std * 255, img.shape)
            img += noise
            img = np.clip(img, 0, 255).astype(np.uint8)
            results['img'] = img
        
        return results

print("Registering RandomBlur in the TRANSFORMS registry...")

@TRANSFORMS.register_module(force=True)
class RandomBlur:
    """Apply random blur to images.
    
    Args:
        blur_kernel_size (tuple): Range for blur kernel size.
        prob (float): Probability of applying blur.
    """
    
    def __init__(self, blur_kernel_size=(3, 7), prob=0.2):
        self.blur_kernel_size = blur_kernel_size
        self.prob = prob
    
    def __call__(self, results: dict) -> dict:
        """Apply random blur to image."""
        return self.transform(results)
    
    def transform(self, results: dict) -> dict:
        """Apply random blur to image."""
        if random.random() < self.prob:
            img = results['img']
            # Ensure kernel size is odd
            ksize = random.randrange(self.blur_kernel_size[0], self.blur_kernel_size[1] + 1, 2)
            if ksize % 2 == 0:
                ksize += 1
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
            results['img'] = img
        
        return results

print("Registering ElasticTransform in the TRANSFORMS registry...")

@TRANSFORMS.register_module(force=True)
class ElasticTransform:
    """Apply elastic deformation to images and keypoints.
    
    Args:
        alpha (float): Displacement strength.
        sigma (float): Smoothness of displacement.
        prob (float): Probability of applying transform.
    """
    
    def __init__(self, alpha=50, sigma=5, prob=0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.prob = prob
    
    def __call__(self, results: dict) -> dict:
        """Apply elastic deformation."""
        return self.transform(results)
    
    def transform(self, results: dict) -> dict:
        """Apply elastic deformation."""
        if random.random() < self.prob:
            img = results['img']
            h, w = img.shape[:2]
            
            # Generate random displacement fields
            dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), 
                                self.sigma, mode="constant", cval=0) * self.alpha
            dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), 
                                self.sigma, mode="constant", cval=0) * self.alpha
            
            # Create coordinate grids
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            x_new = np.clip(x + dx, 0, w - 1)
            y_new = np.clip(y + dy, 0, h - 1)
            
            # Apply deformation to image
            if len(img.shape) == 3:
                warped_img = np.zeros_like(img)
                for c in range(img.shape[2]):
                    warped_img[:, :, c] = cv2.remap(
                        img[:, :, c], x_new.astype(np.float32), y_new.astype(np.float32),
                        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
                    )
            else:
                warped_img = cv2.remap(
                    img, x_new.astype(np.float32), y_new.astype(np.float32),
                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
                )
            
            results['img'] = warped_img
            
            # Apply same deformation to keypoints if they exist
            if 'keypoints' in results:
                keypoints = results['keypoints']
                if keypoints.shape[0] > 0 and keypoints.shape[1] >= 2:
                    for i in range(keypoints.shape[0]):
                        x_coord, y_coord = keypoints[i, 0], keypoints[i, 1]
                        if 0 <= x_coord < w and 0 <= y_coord < h:
                            x_idx = min(int(x_coord), w - 1)
                            y_idx = min(int(y_coord), h - 1)
                            new_x = x_coord + dx[y_idx, x_idx]
                            new_y = y_coord + dy[y_idx, x_idx]
                            keypoints[i, 0] = np.clip(new_x, 0, w - 1)
                            keypoints[i, 1] = np.clip(new_y, 0, h - 1)
                    results['keypoints'] = keypoints
        
        return results

# Print confirmation that the module has been loaded and registration attempted
print(f"LoadImageNumpy registered: {TRANSFORMS.get('LoadImageNumpy') is not None}")
print(f"PhotoMetricDistortion registered: {TRANSFORMS.get('PhotoMetricDistortion') is not None}")
print(f"RandomGaussianNoise registered: {TRANSFORMS.get('RandomGaussianNoise') is not None}")
print(f"RandomBlur registered: {TRANSFORMS.get('RandomBlur') is not None}")
print(f"ElasticTransform registered: {TRANSFORMS.get('ElasticTransform') is not None}")

# This will output all registered transforms for debugging
print("Registered transforms:", list(TRANSFORMS.module_dict.keys())) 