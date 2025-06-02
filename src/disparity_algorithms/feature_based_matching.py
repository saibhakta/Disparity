import cv2
import numpy as np
from .base_disparity import BaseDisparityMatcher

class FeatureBasedMatcher(BaseDisparityMatcher):
    def __init__(self, feature_detector_name="ORB", matcher_name="BFMatcher", num_features=500):
        super().__init__()
        # The actual feature matching algorithm depends on the choice here.
        # Proposal parameter: "Depends on the feature matching algorithm"
        self.feature_detector_name = feature_detector_name
        self.matcher_name = matcher_name
        self.num_features = num_features # Max features for ORB/SIFT etc.

        self.detector = self._create_detector()
        self.matcher = self._create_matcher()
        
        print(f"Warning: {self.get_name()} is a placeholder. Disparity map generation from sparse features is non-trivial.")


    def _create_detector(self):
        if self.feature_detector_name.upper() == "ORB":
            return cv2.ORB_create(nfeatures=self.num_features)
        elif self.feature_detector_name.upper() == "SIFT":
            return cv2.SIFT_create(nfeatures=self.num_features)
        # Add more detectors (e.g., AKAZE) if needed
        else:
            raise ValueError(f"Unsupported feature detector: {self.feature_detector_name}")

    def _create_matcher(self):
        if self.matcher_name == "BFMatcher":
            norm_type = cv2.NORM_HAMMING if self.feature_detector_name.upper() == "ORB" else cv2.NORM_L2
            return cv2.BFMatcher(norm_type, crossCheck=True)
        # Add more matchers (e.g., FlannBasedMatcher)
        else:
            raise ValueError(f"Unsupported matcher: {self.matcher_name}")
            
    def set_params(self, **params):
        super().set_params(**params)
        if "feature_detector_name" in params or "num_features" in params:
            self.detector = self._create_detector()
        if "matcher_name" in params: # Or if detector changes requiring different norm type
            self.matcher = self._create_matcher()


    def get_params(self) -> dict:
        return {
            "feature_detector_name": self.feature_detector_name,
            "matcher_name": self.matcher_name,
            "num_features": self.num_features
        }

    def compute_disparity(self, left_roi_image: np.ndarray, right_roi_image: np.ndarray) -> np.ndarray:
        # 1. Detect keypoints and compute descriptors.
        # 2. Match descriptors.
        # 3. Filter matches (e.g., Lowe's ratio test, RANSAC with Fundamental Matrix if not rectified).
        # 4. For good matches, disparity = kp_left.pt[0] - kp_right.pt[0].
        # 5. This results in a sparse disparity map. Interpolation might be needed for a dense map.

        # This is a simplified dummy version that returns a sparse-like map.
        print(f"Executing dummy {self.get_name()}...")
        
        if len(left_roi_image.shape) == 3:
            gray_left = cv2.cvtColor(left_roi_image, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = left_roi_image
            gray_right = right_roi_image
        
        h, w = gray_left.shape

        kp_left, des_left = self.detector.detectAndCompute(gray_left, None)
        kp_right, des_right = self.detector.detectAndCompute(gray_right, None)

        disparity_map = np.full((h, w), -1.0, dtype=np.float32) # -1 for no disparity

        if des_left is not None and des_right is not None and len(des_left) > 0 and len(des_right) > 0:
            matches = self.matcher.match(des_left, des_right)
            matches = sorted(matches, key=lambda x: x.distance) # Good for BFMatcher with crossCheck

            for match in matches[:min(len(matches), 50)]: # Take some top matches
                pt_left = kp_left[match.queryIdx].pt
                pt_right = kp_right[match.trainIdx].pt

                # Basic epipolar constraint check (assuming rectified images, y-coordinates should be similar)
                if abs(pt_left[1] - pt_right[1]) < 5: # Tolerance of 5 pixels
                    disp = pt_left[0] - pt_right[0]
                    if disp > 0: # Valid disparity
                        y_coord, x_coord = int(round(pt_left[1])), int(round(pt_left[0]))
                        if 0 <= y_coord < h and 0 <= x_coord < w:
                             disparity_map[y_coord, x_coord] = disp
        else:
            print(f"{self.get_name()}: Not enough descriptors found.")

        return disparity_map # This is a sparse map

    def get_default_param_ranges(self) -> dict:
        return {
            'feature_detector_name': ["ORB", "SIFT"],
            'num_features': [200, 500, 1000],
            # Matcher specific params could be added if FLANN is used etc.
        }