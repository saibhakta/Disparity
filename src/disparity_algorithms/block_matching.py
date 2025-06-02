import cv2
import numpy as np
from .base_disparity import BaseDisparityMatcher

class StereoBlockMatcher(BaseDisparityMatcher):
    def __init__(self, numDisparities=64, blockSize=15):
        super().__init__()
        self.numDisparities = numDisparities
        self.blockSize = blockSize
        self.stereo = cv2.StereoBM_create(numDisparities=self.numDisparities, blockSize=self.blockSize)
        # Tunable parameters mentioned in proposal: Window size, Disparity range, Uniqueness threshold
        # cv2.StereoBM parameters:
        # numDisparities: Must be divisible by 16.
        # blockSize: Must be odd.
        # Other params: uniquenessRatio, speckleWindowSize, speckleRange, disp12MaxDiff
        # For simplicity, starting with numDisparities and blockSize.

    def _update_stereo_object(self):
        # Ensure parameters are valid before creating/updating
        self.numDisparities = max(16, self.numDisparities)
        if self.numDisparities % 16 != 0:
            self.numDisparities = ((self.numDisparities // 16) + 1) * 16
        
        self.blockSize = max(5, self.blockSize)
        if self.blockSize % 2 == 0:
            self.blockSize += 1
            
        self.stereo = cv2.StereoBM_create(numDisparities=self.numDisparities, blockSize=self.blockSize)

    def set_params(self, **params):
        super().set_params(**params)
        self._update_stereo_object() # Re-create or update the stereo object with new params

    def get_params(self) -> dict:
        return {
            "numDisparities": self.numDisparities,
            "blockSize": self.blockSize,
            # "uniquenessRatio": self.stereo.getUniquenessRatio(), # Example
        }

    def compute_disparity(self, left_roi_image: np.ndarray, right_roi_image: np.ndarray) -> np.ndarray:
        # StereoBM expects grayscale images
        if len(left_roi_image.shape) == 3:
            gray_left = cv2.cvtColor(left_roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = left_roi_image
        
        if len(right_roi_image.shape) == 3:
            gray_right = cv2.cvtColor(right_roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_right = right_roi_image

        disparity = self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        return disparity

    def get_default_param_ranges(self) -> dict:
        return {
            'numDisparities': [16, 32, 48, 64, 80, 96, 112, 128],
            'blockSize': [5, 7, 9, 11, 13, 15, 17, 19, 21]
        }