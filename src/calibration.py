import cv2
import numpy as np
import glob
import os
import argparse
from utils import display_image # For sanity check

# Define checkerboard parameters
DEFAULT_CHECKERBOARD_SIZE = (5, 8)  # Inner corners (width, height) - As per your camera_calibration.py (5,8) becomes (8,5) for findChessboardCorners if width first
DEFAULT_SQUARE_SIZE = 0.0235  # Square size in meters

def calibrate_stereo_camera(images_dir, output_file, checkerboard_size, square_size, show_undistorted=False):
    """
    Performs stereo camera calibration using checkerboard images.

    Args:
        images_dir (str): Path to the directory containing 'left' and 'right' subdirectories with calibration images.
        output_file (str): Path to save the .npz file with calibration parameters.
        checkerboard_size (tuple): (width, height) of inner corners of the checkerboard.
        square_size (float): Side length of a checkerboard square in meters.
        show_undistorted (bool): If True, displays undistorted images for sanity check.
    """
    print(f"Starting stereo calibration...")
    print(f"Checkerboard size: {checkerboard_size}, Square size: {square_size}m")

    # Prepare object points (3D points in real-world space)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Scale by square size

    # Lists to store object points and image points
    objpoints = []  # 3D points
    imgpoints_left = []  # 2D points from left images
    imgpoints_right = []  # 2D points from right images

    left_images_path = os.path.join(images_dir, "left", "*.jpg") # Assuming jpg, adjust if needed
    right_images_path = os.path.join(images_dir, "right", "*.jpg")

    left_images = sorted(glob.glob(left_images_path))
    right_images = sorted(glob.glob(right_images_path))

    if not left_images or not right_images:
        print(f"Error: No images found in {os.path.join(images_dir, 'left')} or {os.path.join(images_dir, 'right')}")
        print(f"Searched for: {left_images_path} and {right_images_path}")
        return False
    
    if len(left_images) != len(right_images):
        print(f"Error: Mismatched number of left ({len(left_images)}) and right ({len(right_images)}) images.")
        return False

    print(f"Found {len(left_images)} image pairs for calibration.")
    
    img_shape = None

    for i, (left_img_path, right_img_path) in enumerate(zip(left_images, right_images)):
        print(f"Processing pair {i+1}/{len(left_images)}: {os.path.basename(left_img_path)}, {os.path.basename(right_img_path)}")
        img_left = cv2.imread(left_img_path)
        img_right = cv2.imread(right_img_path)
        
        if img_left is None or img_right is None:
            print(f"Warning: Could not read {left_img_path} or {right_img_path}. Skipping.")
            continue

        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        if img_shape is None:
            img_shape = gray_left.shape[::-1] # (width, height)

        # Find checkerboard corners
        # Note: findChessboardCorners expects (corners_cols, corners_rows) which is (checkerboard_width, checkerboard_height)
        ret_l, corners_l = cv2.findChessboardCorners(gray_left, checkerboard_size, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_right, checkerboard_size, None)

        if ret_l and ret_r:
            objpoints.append(objp)
            # Refine corner locations
            cv2.cornerSubPix(gray_left, corners_l, (11, 11), (-1, -1), 
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints_left.append(corners_l)
            
            cv2.cornerSubPix(gray_right, corners_r, (11, 11), (-1, -1), 
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints_right.append(corners_r)

            # Optionally, draw and display corners
            cv2.drawChessboardCorners(img_left, checkerboard_size, corners_l, ret_l)
            display_image('Left Corners', img_left, 1)
            cv2.drawChessboardCorners(img_right, checkerboard_size, corners_r, ret_r)
            display_image('Right Corners', img_right, 1)
        else:
            print(f"Checkerboard not found in one or both images: {os.path.basename(left_img_path)}, {os.path.basename(right_img_path)} (Left: {ret_l}, Right: {ret_r})")

    if not objpoints:
        print("Error: No valid checkerboard pairs found. Calibration failed.")
        return False

    print(f"Using {len(objpoints)} valid image pairs for calibration.")

    # Calibrate each camera separately (initial guess)
    print("Calibrating left camera individually...")
    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_left, img_shape, None, None)
    print("Calibrating right camera individually...")
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_right, img_shape, None, None)

    # Stereo calibration
    print("Performing stereo calibration...")
    flags = cv2.CALIB_FIX_INTRINSIC # If individual calibrations are good
    # flags = 0 # Or try to refine intrinsics as well
    # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    # flags |= cv2.CALIB_RATIONAL_MODEL # Consider if distortion is complex

    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5) # Increased iterations

    ret_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_l, dist_l, mtx_r, dist_r, img_shape,
        criteria=criteria_stereo,
        flags=flags
    )
    
    print(f"Stereo calibration reprojection error: {ret_stereo}") # This is the overall RMS error

    # Stereo rectification
    print("Performing stereo rectification...")
    # R1, R2: Output 3x3 rectification transform (rotation matrix) for the first and second camera.
    # P1, P2: Output 3x4 projection matrix in the new (rectified) coordinate systems for the first and second camera.
    # Q: Output 4x4 disparity-to-depth mapping matrix.
    R1, R2, P1, P2, Q, roi_l, roi_r = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, img_shape, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1 # alpha=-1 lets OpenCV choose scaling
    )

    # Save calibration data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez(output_file, mtx_l=mtx_l, dist_l=dist_l, mtx_r=mtx_r, dist_r=dist_r,
             R=R, T=T, E=E, F=F, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, image_shape=img_shape)
    print(f"Stereo Calibration Complete. Parameters saved to {output_file}")
    print(f"Baseline (Tx): {T[0,0]} meters")

    if show_undistorted and len(left_images) > 0:
        print("Displaying sample undistorted and rectified images. Press any key to continue...")
        from utils import CalibrationData # Local import for this optional block
        calib_data = CalibrationData(output_file)
        
        # Pick a sample image pair
        sample_idx = len(left_images) // 2 
        img_l_orig = cv2.imread(left_images[sample_idx])
        img_r_orig = cv2.imread(right_images[sample_idx])

        rect_l, rect_r = calib_data.rectify_image_pair(img_l_orig, img_r_orig)

        # Draw epipolar lines for visualization
        for i in range(0, rect_l.shape[0], 30): # Draw lines every 30 pixels
            cv2.line(rect_l, (0, i), (rect_l.shape[1], i), (0, 255, 0), 1)
            cv2.line(rect_r, (0, i), (rect_r.shape[1], i), (0, 255, 0), 1)

        combined_view = np.hstack((rect_l, rect_r))
        # Resize if too large for screen
        max_w = 1800
        if combined_view.shape[1] > max_w:
            scale = max_w / combined_view.shape[1]
            combined_view = cv2.resize(combined_view, (0,0), fx=scale, fy=scale)
        
        display_image('Rectified Stereo Pair (with epipolar lines)', combined_view)
        cv2.waitKey(1)
        cv2.destroyAllWindows()

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stereo Camera Calibration")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing 'left' and 'right' subdirs with calibration images.")
    parser.add_argument("--output_file", type=str, required=True, help="Output .npz file for calibration parameters.")
    parser.add_argument("--board_width", type=int, default=DEFAULT_CHECKERBOARD_SIZE[0], help="Number of inner corners width-wise on checkerboard.")
    parser.add_argument("--board_height", type=int, default=DEFAULT_CHECKERBOARD_SIZE[1], help="Number of inner corners height-wise on checkerboard.")
    parser.add_argument("--square_size", type=float, default=DEFAULT_SQUARE_SIZE, help="Size of a checkerboard square in meters.")
    parser.add_argument("--show_undistorted", action='store_true', help="Display a sample undistorted image pair after calibration.")
    args = parser.parse_args()

    checkerboard_dims = (args.board_width, args.board_height)
    
    if not os.path.isdir(args.images_dir):
        print(f"Error: Images directory '{args.images_dir}' not found.")
    else:
        calibrate_stereo_camera(args.images_dir, args.output_file, checkerboard_dims, args.square_size, args.show_undistorted)