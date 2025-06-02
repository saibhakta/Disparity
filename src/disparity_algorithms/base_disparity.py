from abc import ABC, abstractmethod
import numpy as np

class BaseDisparityMatcher(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compute_disparity(self, left_roi_image: np.ndarray, right_roi_image: np.ndarray) -> np.ndarray:
        """
        Computes the disparity map for the given stereo ROI images.

        Args:
            left_roi_image: The left camera's region of interest (grayscale or color).
            right_roi_image: The right camera's region of interest (grayscale or color).

        Returns:
            A 2D numpy array representing the disparity map.
            Disparity values are typically floats or integers.
        """
        pass

    def get_name(self) -> str:
        """Returns the name of the algorithm."""
        return self.__class__.__name__

    def get_params(self) -> dict:
        """
        Returns the current tunable parameters of the algorithm.
        To be implemented by subclasses if they have tunable parameters.
        """
        return {}

    def set_params(self, **params) -> None:
        """
        Sets the tunable parameters of the algorithm.
        To be implemented by subclasses.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            # else:
                # print(f"Warning: Parameter {key} not found in {self.get_name()}")

    def get_default_param_ranges(self) -> dict:
        """
        Returns a dictionary of default parameter ranges for tuning.
        Example: {'numDisparities': [16, 32, 64, 128], 'blockSize': [5, 7, 9, 11, 15]}
        To be implemented by subclasses.
        """
        return {}