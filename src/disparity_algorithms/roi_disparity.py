import numpy as np
from .base_disparity import BaseDisparityMatcher

class ROIDisparity(BaseDisparityMatcher):
    def __init__(self):
        super().__init__()
        # This algorithm has no tunable parameters from its perspective for now
        # Parameters might be related to how ROIs are determined, but that's upstream.

    def compute_disparity(self, left_roi_image: np.ndarray, right_roi_image: np.ndarray, left_metadata: dict, right_metadata: dict) -> np.ndarray:
        """
        Computes disparity based on the X-centers of the detection bounding boxes
        provided in the metadata. Ignores the image content itself.

        Args:
            left_roi_image: The left camera's cropped image (ignored by this algorithm).
            right_roi_image: The right camera's cropped image (ignored by this algorithm).
            left_metadata: Metadata for the left image, must contain 'detection_bbox_in_rectified_image'.
            right_metadata: Metadata for the right image, must contain 'detection_bbox_in_rectified_image'.

        Returns:
            A 1x1 numpy array containing the calculated disparity, or an empty array if data is missing.
        """
        # print(f"ROIDisparity: Left metadata keys: {left_metadata.keys()}")
        # print(f"ROIDisparity: Right metadata keys: {right_metadata.keys()}")

        if 'detection_bbox_in_rectified_image' not in left_metadata or \
           'detection_bbox_in_rectified_image' not in right_metadata:
            print("Warning (ROIDisparity): detection_bbox_in_rectified_image not found in metadata.")
            return np.array([[]], dtype=np.float32) # Return empty or specific error indicator

        left_bbox = left_metadata['detection_bbox_in_rectified_image']
        right_bbox = right_metadata['detection_bbox_in_rectified_image']

        if not all(k in left_bbox for k in ('x', 'w')) or \
           not all(k in right_bbox for k in ('x', 'w')):
            print("Warning (ROIDisparity): 'x' or 'w' missing from detection_bbox_in_rectified_image.")
            return np.array([[]], dtype=np.float32)

        try:
            x_center_L = float(left_bbox['x']) + float(left_bbox['w']) / 2.0
            x_center_R = float(right_bbox['x']) + float(right_bbox['w']) / 2.0
            disparity = x_center_L - x_center_R
            # print(f"ROIDisparity: L_center_X={x_center_L}, R_center_X={x_center_R}, Disp={disparity}")
            return np.array([[disparity]], dtype=np.float32)
        except (TypeError, ValueError) as e:
            print(f"Error (ROIDisparity) calculating disparity from bbox values: {e}")
            return np.array([[]], dtype=np.float32)


    def get_params(self) -> dict:
        """This algorithm does not have tunable parameters via set_params."""
        return {}

    def set_params(self, **params) -> None:
        """This algorithm does not have tunable parameters to set."""
        # print(f"ROIDisparity: set_params called with {params}, but no parameters are tunable.")
        pass

    def get_default_param_ranges(self) -> dict:
        """No parameters to tune for ROIDisparity."""
        return {}

# To ensure get_name returns the class name correctly via BaseDisparityMatcher
# no need to override it here unless a different name is desired. 