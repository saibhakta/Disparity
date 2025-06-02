import cv2
import numpy as np
import json
import argparse
import os

# Ensure utils.py is accessible
try:
    from utils import CalibrationData
except ImportError:
    print("Error: Could not import CalibrationData from utils.py.")
    print("Please ensure utils.py is in the same directory or in your PYTHONPATH.")
    exit(1)

# --- Drawing Configuration for OpenCV Visualization ---
POINT_COLOR_CV = (0, 255, 0)  # Green in BGR for OpenCV (annotated points)
POINT_RADIUS_CV = 4          # Radius of the circle for the point
TEXT_COLOR_CV = (0, 230, 230) # Bright Yellow in BGR for text
TEXT_FONT_CV = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE_CV = 0.6
TEXT_THICKNESS_CV = 1
TEXT_OFFSET_X = 5 # Offset for text from point center
TEXT_OFFSET_Y = -5 # Offset for text from point center

INFO_TEXT_COLOR_CV = (255, 100, 0) # Blueish for overall info
INFO_TEXT_SCALE_CV = 0.7
INFO_TEXT_THICKNESS_CV = 2


def draw_annotations_on_image_cv(image_bgr, points_data, side_prefix):
    """
    Draws annotated points on a single image using OpenCV.
    image_bgr: The image (numpy array in BGR format) to draw on.
    points_data: List of point dictionaries from the annotation file.
    side_prefix: "left" or "right", to select the correct coordinates.
    """
    img_with_annotations = image_bgr.copy()
    for i, point_info in enumerate(points_data):
        x_coord_key = f"{side_prefix}_x_center"
        y_coord_key = f"{side_prefix}_y_center"

        if x_coord_key not in point_info or y_coord_key not in point_info:
            print(f"Warning: Missing '{x_coord_key}' or '{y_coord_key}' for point {i+1} in {side_prefix} image. Skipping.")
            continue

        # Coordinates from JSON are pixel centers (floats)
        x_center = point_info[x_coord_key]
        y_center = point_info[y_coord_key]

        # cv2 functions expect integer coordinates for drawing
        pt_cv_center = (int(round(x_center)), int(round(y_center)))
        point_number = i + 1

        # Draw the point marker (circle)
        cv2.circle(img_with_annotations, pt_cv_center, POINT_RADIUS_CV, POINT_COLOR_CV, -1)  # -1 for filled circle

        # Draw the point number
        text_position = (pt_cv_center[0] + TEXT_OFFSET_X, pt_cv_center[1] + TEXT_OFFSET_Y)
        cv2.putText(img_with_annotations, str(point_number), text_position,
                    TEXT_FONT_CV, TEXT_SCALE_CV, TEXT_COLOR_CV, TEXT_THICKNESS_CV, cv2.LINE_AA)
    return img_with_annotations

def main():
    parser = argparse.ArgumentParser(description="Verification tool to display annotated stereo images using OpenCV.")
    parser.add_argument("--left_image", type=str, required=True, help="Path to the raw left image file.")
    parser.add_argument("--right_image", type=str, required=True, help="Path to the raw right image file.")
    parser.add_argument("--annotation_file", type=str, required=True, help="Path to the JSON annotation file for this image pair.")
    parser.add_argument("--calibration_file", type=str, required=True, help="Path to the stereo_calibration.npz file.")
    args = parser.parse_args()

    # 1. Load Calibration Data
    try:
        calibration_data = CalibrationData(args.calibration_file)
        print(f"Successfully loaded calibration data from: {args.calibration_file}")
    except Exception as e:
        print(f"Error loading calibration file '{args.calibration_file}': {e}")
        return

    # 2. Load Raw Images
    img_l_raw = cv2.imread(args.left_image)
    img_r_raw = cv2.imread(args.right_image)

    if img_l_raw is None:
        print(f"Error: Could not load left image from '{args.left_image}'")
        return
    if img_r_raw is None:
        print(f"Error: Could not load right image from '{args.right_image}'")
        return
    print(f"Successfully loaded raw images.")

    # 3. Rectify Images
    try:
        # Ensure rectification maps are initialized for the current image size
        # (width, height) from image shape (height, width, channels)
        current_image_size = (img_l_raw.shape[1], img_l_raw.shape[0])
        if calibration_data.image_size != current_image_size or calibration_data.map1_l is None:
             calibration_data.init_rectification_maps(current_image_size)
        
        rect_l, rect_r = calibration_data.rectify_image_pair(img_l_raw, img_r_raw)
        print("Images rectified successfully.")
    except Exception as e:
        print(f"Error during image rectification: {e}")
        return

    # 4. Load Annotations
    try:
        with open(args.annotation_file, 'r') as f:
            annotation_data = json.load(f)
        
        annotated_points = annotation_data.get("annotated_points", [])
        if not annotated_points:
            print(f"Warning: No 'annotated_points' found in {args.annotation_file}. Displaying rectified images without points.")
        else:
            print(f"Successfully loaded {len(annotated_points)} annotated points from {args.annotation_file}")
            
        avg_disp_str = f"{annotation_data.get('avg_disparity', 'N/A'):.2f}" if isinstance(annotation_data.get('avg_disparity'), float) else str(annotation_data.get('avg_disparity', 'N/A'))
        num_pts_str = str(annotation_data.get('num_points', 'N/A'))
        info_text_content = f"File: {os.path.basename(args.annotation_file)} | Avg Disp: {avg_disp_str}px | Pts: {num_pts_str}"

    except FileNotFoundError:
        print(f"Error: Annotation file not found at '{args.annotation_file}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{args.annotation_file}'")
        return
    except Exception as e:
        print(f"Error loading or parsing annotation file: {e}")
        return

    # 5. Draw Annotations on Rectified Images
    if annotated_points: # Only draw if points exist
        annotated_rect_l = draw_annotations_on_image_cv(rect_l, annotated_points, "left")
        annotated_rect_r = draw_annotations_on_image_cv(rect_r, annotated_points, "right")
    else:
        annotated_rect_l = rect_l.copy() # Use copies to avoid modifying original rectified images
        annotated_rect_r = rect_r.copy()

    # 6. Combine and Display
    # Ensure images have the same height for hconcat
    if annotated_rect_l.shape[0] != annotated_rect_r.shape[0]:
        print("Warning: Rectified left and right images have different heights. This should not happen with proper calibration.")
        # As a fallback, try to resize the smaller one, or display separately. For now, display as is.
        # For simplicity in this verification tool, we'll proceed, but this indicates a potential issue.
    
    # Add overall info text to the top of the left image
    cv2.putText(annotated_rect_l, info_text_content, (10, 30),
                TEXT_FONT_CV, INFO_TEXT_SCALE_CV, INFO_TEXT_COLOR_CV, INFO_TEXT_THICKNESS_CV, cv2.LINE_AA)

    try:
        combined_image = cv2.hconcat([annotated_rect_l, annotated_rect_r])
    except cv2.error as e:
        print(f"OpenCV error during hconcat (likely due to differing image dimensions/types): {e}")
        print("Displaying images in separate windows as a fallback.")
        cv2.imshow("Annotated Rectified Left", annotated_rect_l)
        cv2.imshow("Annotated Rectified Right", annotated_rect_r)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
        
    window_title = f"Annotation Verification (Press any key to close)"
    cv2.imshow(window_title, combined_image)
    print("Displaying annotated rectified images. Press any key in the OpenCV window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

    # python src/verify_annotations.py \
    # --left_image data/basketball_images/left/left_01.jpg \
    # --right_image data/basketball_images/right/right_01.jpg \
    # --annotation_file data/annotations/1.json \
    # --calibration_file data/calibration.npz