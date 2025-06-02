import cv2
import numpy as np
from .base_disparity import BaseDisparityMatcher

class SemiGlobalBlockMatcher(BaseDisparityMatcher):
    def __init__(self, minDisparity=0, numDisparities=64, blockSize=5, P1=None, P2=None,
                 disp12MaxDiff=1, preFilterCap=63, uniquenessRatio=10,
                 speckleWindowSize=100, speckleRange=32, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY):
        super().__init__()
        self.minDisparity = minDisparity
        self.numDisparities = numDisparities
        self.blockSize = blockSize # Initial block size
        self.disp12MaxDiff = disp12MaxDiff
        self.preFilterCap = preFilterCap
        self.uniquenessRatio = uniquenessRatio
        self.speckleWindowSize = speckleWindowSize
        self.speckleRange = speckleRange
        self.mode = mode
        
        self._num_input_channels = 1 # Default to 1 channel for initial P1/P2 calculation

        self._user_explicitly_set_P1 = (P1 is not None)
        self._user_explicitly_set_P2 = (P2 is not None)

        self.P1 = P1 if self._user_explicitly_set_P1 else (8 * self._num_input_channels * self.blockSize**2)
        self.P2 = P2 if self._user_explicitly_set_P2 else (32 * self._num_input_channels * self.blockSize**2)
        
        self.stereo = None # Will be created by _update_stereo_object
        self._update_stereo_object()

    def _update_stereo_object(self):
        # Ensure parameters are valid before creating/updating stereo object
        current_numDisparities = max(16, self.numDisparities)
        if current_numDisparities % 16 != 0:
            current_numDisparities = ((current_numDisparities // 16) + 1) * 16
        
        current_blockSize = max(1, self.blockSize)
        if current_blockSize % 2 == 0 and current_blockSize > 0:
             current_blockSize += 1
        
        # Use self.P1 and self.P2 which are managed by __init__ and set_params
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=self.minDisparity,
            numDisparities=current_numDisparities,
            blockSize=current_blockSize, # Use the validated current_blockSize
            P1=self.P1,
            P2=self.P2,
            disp12MaxDiff=self.disp12MaxDiff,
            preFilterCap=self.preFilterCap,
            uniquenessRatio=self.uniquenessRatio,
            speckleWindowSize=self.speckleWindowSize,
            speckleRange=self.speckleRange,
            mode=self.mode
        )
    
    def set_params(self, **params):
        super().set_params(**params) # This updates self.blockSize, self.P1, self.P2 etc. directly

        # Check if P1 or P2 were part of this specific params update
        if 'P1' in params:
            self._user_explicitly_set_P1 = True
        if 'P2' in params:
            self._user_explicitly_set_P2 = True
        
        # If blockSize was changed, and P1/P2 are not user-managed, recalculate them.
        if 'blockSize' in params: 
            if not self._user_explicitly_set_P1:
                 self.P1 = 8 * self._num_input_channels * self.blockSize**2
            if not self._user_explicitly_set_P2:
                 self.P2 = 32 * self._num_input_channels * self.blockSize**2
        
        self._update_stereo_object()

    def get_params(self) -> dict:
        # Return the current state of parameters, preferably from the stereo object itself if available
        if self.stereo:
            return {
                "minDisparity": self.stereo.getMinDisparity(),
                "numDisparities": self.stereo.getNumDisparities(),
                "blockSize": self.stereo.getBlockSize(),
                "P1": self.stereo.getP1(),
                "P2": self.stereo.getP2(),
                "disp12MaxDiff": self.stereo.getDisp12MaxDiff(),
                "preFilterCap": self.stereo.getPreFilterCap(),
                "uniquenessRatio": self.stereo.getUniquenessRatio(),
                "speckleWindowSize": self.stereo.getSpeckleWindowSize(),
                "speckleRange": self.stereo.getSpeckleRange(),
                "mode": self.stereo.getMode()
            }
        else: # Fallback if stereo object not created (should not happen with current logic)
            return {
                "minDisparity": self.minDisparity,
                "numDisparities": self.numDisparities,
                "blockSize": self.blockSize,
                "P1": self.P1,
                "P2": self.P2,
                "disp12MaxDiff": self.disp12MaxDiff,
                "preFilterCap": self.preFilterCap,
                "uniquenessRatio": self.uniquenessRatio,
                "speckleWindowSize": self.speckleWindowSize,
                "speckleRange": self.speckleRange,
                "mode": self.mode
            }

    def compute_disparity(self, left_roi_image: np.ndarray, right_roi_image: np.ndarray) -> np.ndarray:
        new_num_input_channels = 1
        if len(left_roi_image.shape) == 2:
             new_num_input_channels = 1
        elif len(left_roi_image.shape) == 3:
             new_num_input_channels = left_roi_image.shape[2]
        else:
            raise ValueError("Unsupported image format")

        if self._num_input_channels != new_num_input_channels:
            self._num_input_channels = new_num_input_channels
            # If P1/P2 are not user-managed, recalculate them based on new channel number.
            if not self._user_explicitly_set_P1:
                 self.P1 = 8 * self._num_input_channels * self.blockSize**2
            if not self._user_explicitly_set_P2:
                 self.P2 = 32 * self._num_input_channels * self.blockSize**2
            self._update_stereo_object() 

        disparity = self.stereo.compute(left_roi_image, right_roi_image).astype(np.float32) / 16.0
        return disparity
        
    def get_default_param_ranges(self) -> dict:
        return {
            'numDisparities': [32, 64, 96, 128, 192, 256],
            'blockSize': [3, 5, 7, 9, 11], 
            'uniquenessRatio': [5, 10, 15],
            'speckleWindowSize': [0, 50, 100, 150, 200],
            'speckleRange': [1, 2, 4, 8, 16, 32],
            'disp12MaxDiff': [-1, 0, 1, 5]
        }