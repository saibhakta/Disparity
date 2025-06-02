import cv2
import numpy as np
import os
import argparse
import json
import glob
from .utils import CalibrationData, load_image, display_image, save_ground_truth_annotation, load_ground_truth_annotation

# Global variables to store points and current image data
points_left = []
points_right = []
current_left_rect = None
current_right_rect = None
current_image_pair_name = ""
annotations_dir_global = ""
calib_data_global = None

def click_event_left(event, x, y, flags, param):
    global points_left, current_left_rect
    if event == cv2.EVENT_LBUTTONDOWN:
        points_left.append((x, y))
        cv2.circle(current_left_rect, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Rectified Left Image (Annotate)", current_left_rect)
        print(f"Left image: Clicked at ({x}, {y})")

def click_event_right(event, x, y, flags, param):
    global points_right, current_right_rect
    if event == cv2.EVENT_LBUTTONDOWN:
        points_right.append((x, y))
        cv2.circle(current_right_rect, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Rectified Right Image (Annotate)", current_right_rect)
        print(f"Right image: Clicked at ({x}, {y})")

def annotate_stereo_pair(left_img_path, right_img_path, calib_data: CalibrationData, annotations_dir):
    global points_left, points_right, current_left_rect, current_right_rect
    global current_image_pair_name, annotations_dir_global, calib_data_global

    current_image_pair_name = os.path.basename(left_img_path).replace("left_", "").replace(".jpg", "")
    annotations_dir_global = annotations_dir
    calib_data_global = calib_data
    
    annotation_file_path = os.path.join(annotations_dir, f"{current_image_pair_name}_annotation.json")
    
    # Check if annotation already exists
    if os.path.exists(annotation_file_path):
        existing_annotation = load_ground_truth_annotation(annotation_file_path)
        if existing_annotation and "avg_disparity" in existing_annotation: # Check for a key field
            print(f"Annotation already exists for {current_image_pair_name}. Avg Disparity: {existing_annotation['avg_disparity']:.2f}. Skipping (S) or Re-annotate (R)?")
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('s'):
                    print("Skipping.")
                    return True # Indicate skip
                elif key == ord('r'):
                    print("Re-annotating.")
                    break # Proceed to annotate
                elif key == ord('q'):
                    print("Quitting.")
                    return False # Indicate quit

    img_left_orig = load_image(left_img_path)
    img_right_orig = load_image(right_img_path)

    if img_left_orig is None or img_right_orig is None:
        return True # Skip if images can't be loaded

    current_left_rect, current_right_rect = calib_data.rectify_image_pair(img_left_orig, img_right_orig)

    # Create copies for drawing without modifying originals used for re-display
    display_left = current_left_rect.copy()
    display_right = current_right_rect.copy()
    
    # Draw epipolar lines for guidance
    for i in range(0, display_left.shape[0], 30):
        cv2.line(display_left, (0, i), (display_left.shape[1], i), (0, 255, 0), 1)
        cv2.line(display_right, (0, i), (display_right.shape[1], i), (0, 255, 0), 1)

    cv2.imshow("Rectified Left Image (Annotate)", display_left)
    cv2.imshow("Rectified Right Image (Annotate)", display_right)
    cv2.setMouseCallback("Rectified Left Image (Annotate)", click_event_left)
    cv2.setMouseCallback("Rectified Right Image (Annotate)", click_event_right)

    print(f"\nAnnotating: {current_image_pair_name}")
    print("Click corresponding points on the basketball in the left and right images.")
    print("Ensure you click the same number of points in both images, in the same order.")
    print("Press 's' to save current points and move to next image.")
    print("Press 'r' to reset points for the current image.")
    print("Press 'q' to quit.")

    points_left = []
    points_right = []

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'): # Save
            if not points_left or not points_right:
                print("No points selected. Cannot save.")
                continue
            if len(points_left) != len(points_right):
                print("Mismatch in number of points between left and right images. Please reset and re-annotate.")
                continue

            disparities = []
            valid_pairs_for_json = []
            for p_l, p_r in zip(points_left, points_right):
                # Basic check: y-coordinates should be very close in rectified images
                if abs(p_l[1] - p_r[1]) > 10: # Allow some tolerance for manual clicks
                    print(f"Warning: Large Y-discrepancy for points {p_l} and {p_r}. This pair might be invalid.")
                disparity = float(p_l[0] - p_r[0])
                if disparity < 0 : # X_left should generally be >= X_right
                    print(f"Warning: Negative disparity ({disparity}) for points {p_l} and {p_r}. This might be an incorrect match.")
                disparities.append(disparity)
                valid_pairs_for_json.append({'left_x': p_l[0], 'left_y': p_l[1], 'right_x': p_r[0], 'right_y': p_r[1], 'disparity': disparity})


            if disparities:
                avg_disparity = np.mean(disparities)
                std_dev_disparity = np.std(disparities)
                print(f"Average Disparity: {avg_disparity:.2f} pixels")
                print(f"Std Dev Disparity: {std_dev_disparity:.2f} pixels")

                annotation_data = {
                    "image_pair_name": current_image_pair_name,
                    "left_image": os.path.basename(left_img_path),
                    "right_image": os.path.basename(right_img_path),
                    "annotated_points": valid_pairs_for_json,
                    "avg_disparity": float(avg_disparity),
                    "std_dev_disparity": float(std_dev_disparity),
                    "num_points": len(points_left)
                }
                save_ground_truth_annotation(annotation_file_path, annotation_data)
                print(f"Annotation saved to {annotation_file_path}")
            else:
                print("No valid disparities calculated. Not saving.")
            break # Move to next image

        elif key == ord('r'): # Reset
            print("Resetting points for current image pair.")
            points_left = []
            points_right = []
            # Redraw the original rectified images without points
            current_left_rect, current_right_rect = calib_data_global.rectify_image_pair(load_image(left_img_path), load_image(right_img_path)) # Re-rectify to get clean copy
            display_left_reset = current_left_rect.copy()
            display_right_reset = current_right_rect.copy()
            for i in range(0, display_left_reset.shape[0], 30):
                cv2.line(display_left_reset, (0, i), (display_left_reset.shape[1], i), (0, 255, 0), 1)
                cv2.line(display_right_reset, (0, i), (display_right_reset.shape[1], i), (0, 255, 0), 1)

            cv2.imshow("Rectified Left Image (Annotate)", display_left_reset)
            cv2.imshow("Rectified Right Image (Annotate)", display_right_reset)


        elif key == ord('q'): # Quit
            print("Quitting annotation tool.")
            return False # Indicate quit
    
    return True # Indicate success/continue

def main_annotation_tool(images_dir, calibration_file, annotations_dir):
    calib = CalibrationData(calibration_file)

    left_image_files = sorted(glob.glob(os.path.join(images_dir, "left", "*.jpg")))
    right_image_files = sorted(glob.glob(os.path.join(images_dir, "right", "*.jpg")))

    if not left_image_files or not right_image_files:
        print(f"No JPG images found in {os.path.join(images_dir, 'left')} or {os.path.join(images_dir, 'right')}")
        return
    
    if len(left_image_files) != len(right_image_files):
        print("Warning: Mismatch in number of left and right images. Processing common subset based on sorted names.")
        # This simple pairing might not be robust if filenames are not perfectly matched.
        # A more robust way would be to parse timestamps from filenames if they exist.

    os.makedirs(annotations_dir, exist_ok=True)
    
    for i in range(len(left_image_files)):
        left_path = left_image_files[i]
        # Try to find corresponding right image (simple name-based matching)
        base_name_left = os.path.basename(left_path)
        expected_right_name = base_name_left.replace("left_", "right_")
        right_path = os.path.join(images_dir, "right", expected_right_name)

        if not os.path.exists(right_path):
            print(f"Could not find matching right image for {left_path} (expected {expected_right_name}). Skipping.")
            continue
            
        if not annotate_stereo_pair(left_path, right_path, calib, annotations_dir):
            break # User chose to quit

    cv2.destroyAllWindows()
    print("Annotation process finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ground Truth Disparity Annotation Tool")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing 'left' and 'right' subdirs with raw stereo pairs.")
    parser.add_argument("--calibration_file", type=str, required=True, help="Path to the stereo_calibration.npz file.")
    parser.add_argument("--annotations_dir", type=str, required=True, help="Directory to save ground truth JSON annotation files.")
    args = parser.parse_args()

    main_annotation_tool(args.images_dir, args.calibration_file, args.annotations_dir)