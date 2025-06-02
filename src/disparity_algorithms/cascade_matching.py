import cv2
import numpy as np
from .base_disparity import BaseDisparityMatcher
from .sgbm import SemiGlobalBlockMatcher # Example: Use SGBM as the base matcher

class CascadeMatcher(BaseDisparityMatcher):
    def __init__(self, num_pyramid_levels=3, scale_step_between_levels=0.5, window_size=7, base_matcher_class=SemiGlobalBlockMatcher):
        super().__init__()
        self.num_pyramid_levels = num_pyramid_levels
        self.scale_step_between_levels = scale_step_between_levels # e.g. 0.5 for halving size each level
        self.window_size = window_size # This might be passed to the base_matcher
        self.base_matcher_class = base_matcher_class
        # Instantiate a base matcher (its params might need to be exposed or tuned)
        self.base_matcher_instance = self.base_matcher_class(blockSize=self.window_size)

        print(f"Warning: {self.get_name()} is a placeholder and its current implementation is basic.")

    def set_params(self, **params):
        super().set_params(**params)
        if 'window_size' in params or 'base_matcher_params' in params:
            base_params = params.get('base_matcher_params', {})
            if 'window_size' in params : # Allow overriding base matcher's blocksize
                base_params['blockSize'] = self.window_size
            self.base_matcher_instance = self.base_matcher_class(**base_params)


    def get_params(self) -> dict:
        return {
            "num_pyramid_levels": self.num_pyramid_levels,
            "scale_step_between_levels": self.scale_step_between_levels,
            "window_size": self.window_size, # Or self.base_matcher_instance.get_params()['blockSize']
            "base_matcher_name": self.base_matcher_class.__name__,
            "base_matcher_params": self.base_matcher_instance.get_params()
        }

    def compute_disparity(self, left_roi_image: np.ndarray, right_roi_image: np.ndarray) -> np.ndarray:
        # Basic cascade matching (conceptual)
        # 1. Build image pyramids for left and right images.
        # 2. Start at coarsest level:
        #    - Compute disparity.
        #    - Upscale disparity map to next finer level.
        # 3. At finer levels:
        #    - Use upscaled disparity from coarser level to refine search range.
        #    - Compute disparity within this refined range.
        #    - Upscale.
        # 4. Repeat until finest level.

        # This is a simplified dummy version. A real implementation is more involved.
        print(f"Executing dummy {self.get_name()} using {self.base_matcher_class.__name__}...")

        # For now, just run the base matcher on the original resolution
        if len(left_roi_image.shape) == 3:
            current_left = cv2.cvtColor(left_roi_image, cv2.COLOR_BGR2GRAY)
            current_right = cv2.cvtColor(right_roi_image, cv2.COLOR_BGR2GRAY)
        else:
            current_left = left_roi_image.copy()
            current_right = right_roi_image.copy()

        # Simple dummy: just use the base matcher at full resolution
        disparity_map = self.base_matcher_instance.compute_disparity(current_left, current_right)
        
        return disparity_map

    def get_default_param_ranges(self) -> dict:
        # These params are for the cascade logic itself.
        # Tuning also involves the base_matcher's parameters.
        return {
            'num_pyramid_levels': [2, 3, 4],
            'scale_step_between_levels': [0.5, 0.75], # Downscale factor
            'window_size': [5, 7, 9, 11] # Example, forwarded to base matcher's blocksize
        }