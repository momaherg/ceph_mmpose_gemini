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

        results['img_shape'] = img.shape[:2] # (h, w)
        results['ori_shape'] = img.shape[:2] # (h, w)
        return results

print(f"[DEBUG custom_transforms.py] Attempting to register LoadImageNumpy.")
print(f"[DEBUG custom_transforms.py] TRANSFORMS registry object: {TRANSFORMS}")
if TRANSFORMS is not None and hasattr(TRANSFORMS, 'scope'):
    print(f"[DEBUG custom_transforms.py] TRANSFORMS scope: {TRANSFORMS.scope}")
    print(f"[DEBUG custom_transforms.py] Is 'LoadImageNumpy' in TRANSFORMS after registration attempt? {'LoadImageNumpy' in TRANSFORMS}")
else:
    print(f"[DEBUG custom_transforms.py] TRANSFORMS object is None or has no scope attribute.") 