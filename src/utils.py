import cv2
import numpy as np
import os
import json

class CalibrationData:
    def __init__(self, filepath):
        try:
            calib_data = np.load(filepath)
            self.mtx_l = calib_data['mtx_l']
            self.dist_l = calib_data['dist_l']
            self.mtx_r = calib_data['mtx_r']
            self.dist_r = calib_data['dist_r']
            self.R = calib_data['R']
            self.T = calib_data['T']
            self.R1 = calib_data['R1']
            self.R2 = calib_data['R2']
            self.P1 = calib_data['P1']
            self.P2 = calib_data['P2']
            self.Q = calib_data['Q']
            self.image_size = None # Will be set during map initialization
            self.map1_l, self.map2_l = None, None
            self.map1_r, self.map2_r = None, None
            print(f"Calibration data loaded from {filepath}")
        except Exception as e:
            print(f"Error loading calibration data from {filepath}: {e}")
            raise

    def init_rectification_maps(self, image_size):
        if self.image_size == image_size and self.map1_l is not None:
            return # Already initialized for this size
            
        self.image_size = image_size
        self.map1_l, self.map2_l = cv2.initUndistortRectifyMap(
            self.mtx_l, self.dist_l, self.R1, self.P1,
            self.image_size, cv2.CV_16SC2
        )
        self.map1_r, self.map2_r = cv2.initUndistortRectifyMap(
            self.mtx_r, self.dist_r, self.R2, self.P2,
            self.image_size, cv2.CV_16SC2
        )
        print(f"Rectification maps initialized for image size: {image_size}")

    def rectify_image_pair(self, img_left, img_right):
        if self.map1_l is None or self.image_size != img_left.shape[1::-1]: # (width, height)
            self.init_rectification_maps(img_left.shape[1::-1])
        
        rect_left = cv2.remap(img_left, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        rect_right = cv2.remap(img_right, self.map1_r, self.map2_r, cv2.INTER_LINEAR)
        return rect_left, rect_right

def load_image(image_path, grayscale=False):
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(image_path, flag)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return img

def save_image(image_path, image):
    cv2.imwrite(image_path, image)

def display_image(window_name, image, wait_key_ms=0):
    cv2.imshow(window_name, image)
    return cv2.waitKey(wait_key_ms)

def draw_roi(image, roi_xywh, color=(0, 255, 0), thickness=2):
    """ roi_xywh: (x, y, width, height) """
    x, y, w, h = map(int, roi_xywh)
    img_copy = image.copy()
    cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, thickness)
    return img_copy

def crop_to_roi(image, roi_xywh):
    """ roi_xywh: (x, y, width, height) """
    x, y, w, h = map(int, roi_xywh)
    return image[y:y+h, x:x+w]

def load_ground_truth_annotation(filepath: str) -> dict:
    """Loads ground truth data from a JSON file."""
    if not os.path.exists(filepath):
        print(f"Warning: Ground truth file not found: {filepath}")
        return {}
    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {filepath}")
            return {}

def save_ground_truth_annotation(filepath: str, data: dict):
    """Saves ground truth data to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)