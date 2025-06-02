import cv2
import numpy as np
import os
import time
import json
import pandas as pd
import glob
from .utils import CalibrationData, load_image, load_ground_truth_annotation, crop_to_roi, draw_roi
from .yolo_detector import YoloDetector # Assuming YoloDetector is in the same directory or Python path
from .disparity_algorithms import AVAILABLE_ALGORITHMS, BaseDisparityMatcher

class StereoEvaluator:
    def __init__(self, calibration_file_path: str, yolo_hef_path: str, yolo_so_path: str, 
                 target_class_id=0, yolo_min_confidence=0.3):
        self.calib_data = CalibrationData(calibration_file_path)
        self.yolo_detector = None
        if YoloDetector and yolo_hef_path: # Check if YoloDetector class exists
            try:
                self.yolo_detector = YoloDetector(
                    hef_path=yolo_hef_path,
                    postprocess_so_path=yolo_so_path,
                    target_class_id=target_class_id,
                    min_confidence=yolo_min_confidence
                )
                self.yolo_detector.start()
            except Exception as e:
                print(f"Error initializing YOLO detector: {e}. Proceeding without YOLO ROI.")
                self.yolo_detector = None
        else:
            print("YOLO detector parameters not provided or YoloDetector class unavailable. Proceeding without YOLO ROI.")

        self.results = [] # List to store dicts of results

    def _get_basketball_roi(self, rectified_left_image: np.ndarray, rectified_right_image: np.ndarray):
        """
        Detects basketball ROI in the left image and returns a common ROI for both.
        For simplicity, this version uses ROI from left and applies similar region to right.
        A more advanced version could try to find ROI in both and merge/validate.
        """
        if not self.yolo_detector or not self.yolo_detector._running:
            print("YOLO detector not available or not running. Cannot get ROI.")
            # Fallback: use the whole image as ROI if YOLO fails
            h, w = rectified_left_image.shape[:2]
            return (0, 0, w, h), (0,0,w,h)

        # YOLO expects RGB. OpenCV loads BGR by default.
        # Assuming rectified_left_image is BGR
        img_rgb = cv2.cvtColor(rectified_left_image, cv2.COLOR_BGR2RGB)
        
        roi_xywh_left = self.yolo_detector.detect_basketball_roi(img_rgb)

        if roi_xywh_left:
            # For the right image, we can assume the basketball is roughly at the same y-position
            # and x-position will be shifted due to disparity.
            # A simple approach is to use the same ROI window shifted, or a slightly larger one.
            # For now, let's try to use a similar ROI on the right image, centered around the same y.
            # The width of the ROI on the right image should account for the object and some margin.
            
            # Simple approach: use the same ROI for both. This assumes the object doesn't move too much horizontally
            # relative to the ROI window itself due to disparity, or the ROI is large enough.
            # This is a simplification. True stereo ROI requires matching.
            # For evaluation, we are cropping *before* disparity calculation.
            
            # Ensure ROI is within bounds
            x, y, w, h = roi_xywh_left
            img_h, img_w = rectified_left_image.shape[:2]
            
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            bounded_roi_left = (x,y,w,h)
            
            # For the right image, we can try to use the same y, h.
            # The x, w might need adjustment if the disparity is very large and ROI is tight.
            # For now, assume the same ROI box dimensions work for both, or make right ROI a bit wider.
            # This is a simplification; true alignment requires disparity.
            # The proposal says "ROI is then cropped to focus the stereo matcher on the ball alone."
            # This implies a single ROI after detection.
            
            # Use the same ROI for both, taken from the left image's detection.
            bounded_roi_right = bounded_roi_left 
            
            return bounded_roi_left, bounded_roi_right
        else:
            print("YOLO did not detect basketball. Using full image as ROI.")
            h, w = rectified_left_image.shape[:2]
            return (0, 0, w, h), (0,0,w,h)


    def evaluate_pair(self, left_img_path: str, right_img_path: str, ground_truth_disp_val: float,
                      disparity_matcher: BaseDisparityMatcher, debug_display=False):
        """
        Evaluates a single stereo pair with a given disparity matching algorithm.
        Args:
            left_img_path, right_img_path: Paths to the stereo images.
            ground_truth_disp_val: Single float value for average disparity of the basketball.
            disparity_matcher: An instance of a BaseDisparityMatcher subclass.
            debug_display: If True, shows intermediate images and results.
        Returns:
            A dictionary with evaluation metrics for this pair and algorithm.
        """
        try:
            img_left_orig = load_image(left_img_path)
            img_right_orig = load_image(right_img_path)
        except FileNotFoundError as e:
            print(e)
            return None

        # 1. Undistort and Rectify
        rect_left, rect_right = self.calib_data.rectify_image_pair(img_left_orig, img_right_orig)

        if debug_display:
            cv2.imshow("Rectified Left", rect_left)
            cv2.imshow("Rectified Right", rect_right)
            if cv2.waitKey(100) & 0xFF == ord('q'): return "quit"


        # 2. Get ROI of the basketball (using YOLO on left rectified image)
        # This returns ROI for left and a corresponding one for right.
        # Proposal: "Each image pair will pass through a YOLOv11-n network that locates the basketball
        # and returns a region of interest (ROI); the ROI is then cropped to focus the stereo matcher on the ball alone."
        # This implies a single ROI. The yolo_detector gives one ROI.
        
        roi_xywh_left, _ = self._get_basketball_roi(rect_left, rect_right) # Use the one from left for both crops

        if roi_xywh_left is None or roi_xywh_left[2] == 0 or roi_xywh_left[3] == 0 : # width or height is zero
            print(f"Invalid ROI detected for {os.path.basename(left_img_path)}. Using full image.")
            h_full, w_full = rect_left.shape[:2]
            roi_xywh_left = (0,0, w_full, h_full)


        # 3. Crop images to ROI
        left_ball_roi_img = crop_to_roi(rect_left, roi_xywh_left)
        right_ball_roi_img = crop_to_roi(rect_right, roi_xywh_left) # Use same ROI for right

        if left_ball_roi_img.size == 0 or right_ball_roi_img.size == 0:
            print(f"ROI resulted in empty image for {os.path.basename(left_img_path)}. Skipping pair for this algorithm.")
            return None


        if debug_display:
            display_left_roi_viz = draw_roi(rect_left.copy(), roi_xywh_left)
            cv2.imshow("Left with ROI", display_left_roi_viz)
            cv2.imshow("Cropped Left ROI", left_ball_roi_img)
            cv2.imshow("Cropped Right ROI", right_ball_roi_img)
            if cv2.waitKey(100) & 0xFF == ord('q'): return "quit"

        # 4. Compute Disparity using the provided algorithm
        algo_name = disparity_matcher.get_name()
        algo_params = disparity_matcher.get_params()

        start_time = time.perf_counter()
        # Some algorithms expect grayscale, some can handle color.
        # The algorithm implementation should handle conversion if needed.
        computed_disparity_map = disparity_matcher.compute_disparity(left_ball_roi_img, right_ball_roi_img)
        runtime_ms = (time.perf_counter() - start_time) * 1000

        if computed_disparity_map is None:
            print(f"Algorithm {algo_name} failed to compute disparity map for {os.path.basename(left_img_path)}")
            return None
        
        # 5. Extract disparity value from the computed map for the basketball region.
        # This is tricky. The ground truth is a single average disparity for the ball.
        # The computed map is dense (or sparse for feature-based).
        # We need a representative disparity from the computed map within the ball's area.
        # Simplification: Take the median/mean of non-zero/valid disparities in the center of the ROI.
        # A more robust way would be to segment the ball within the ROI (e.g. using color, or if YOLO gave a mask)
        # and average disparities only on the ball pixels.
        # For now, let's take a central patch of the computed disparity map.
        
        h_roi, w_roi = computed_disparity_map.shape[:2]
        if h_roi == 0 or w_roi == 0:
            print(f"Computed disparity map is empty for {os.path.basename(left_img_path)} with {algo_name}. Skipping.")
            return None

        # Consider a central region for disparity extraction, e.g., central 50% x 50%
        center_y_start, center_y_end = int(h_roi * 0.25), int(h_roi * 0.75)
        center_x_start, center_x_end = int(w_roi * 0.25), int(w_roi * 0.75)
        
        central_patch_disp = computed_disparity_map[center_y_start:center_y_end, center_x_start:center_x_end]
        
        # Filter out invalid disparity values (e.g., <=0 or placeholder values like -1 from sparse)
        valid_disparities = central_patch_disp[central_patch_disp > 0] 

        if valid_disparities.size > 0:
            avg_computed_disparity = np.median(valid_disparities) # Median is often more robust to outliers
        else:
            avg_computed_disparity = 0 # Or NaN, or handle as error
            print(f"Warning: No valid disparities in central patch of computed map for {os.path.basename(left_img_path)} with {algo_name}.")


        # 6. Calculate Metrics
        # Proposal metrics: Average absolute disparity error (pixels), Standard deviation of disparity error (pixels)
        # Here, we have one ground truth average disparity and one computed average disparity for the ball.
        # So, "average absolute disparity error" for *this pair* is just the absolute error.
        # The "standard deviation of disparity error" would be calculated over all image pairs later.
        
        absolute_error = abs(avg_computed_disparity - ground_truth_disp_val)

        if debug_display:
            print(f"GT Disparity: {ground_truth_disp_val:.2f}, Computed Avg Disp: {avg_computed_disparity:.2f}, Abs Error: {absolute_error:.2f}")
            disp_map_display = cv2.normalize(computed_disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imshow(f"Computed Disparity ({algo_name})", disp_map_display)
            if cv2.waitKey(0) & 0xFF == ord('q'): return "quit" # Wait indefinitely for this one

        return {
            "left_image": os.path.basename(left_img_path),
            "right_image": os.path.basename(right_img_path),
            "algorithm": algo_name,
            "params": json.dumps(algo_params), # Store params for reproducibility
            "ground_truth_disparity": ground_truth_disp_val,
            "computed_avg_disparity_roi": avg_computed_disparity,
            "absolute_error_roi": absolute_error,
            "runtime_ms": runtime_ms
        }

    def run_evaluation(self, data_dir: str, annotations_dir: str, output_dir: str,
                       algorithms_to_test: list = None, debug_each_step=False):
        """
        Runs the full evaluation pipeline.
        Args:
            data_dir: Directory with 'left' and 'right' subdirs of raw stereo pairs.
            annotations_dir: Directory with ground truth JSON files.
            output_dir: Directory to save evaluation results (e.g., CSV).
            algorithms_to_test: List of algorithm names (keys from AVAILABLE_ALGORITHMS). If None, tests all.
            debug_each_step: If True, shows intermediate images and prompts for each step.
        """
        os.makedirs(output_dir, exist_ok=True)

        left_image_files = sorted(glob.glob(os.path.join(data_dir, "left", "*.jpg")))
        
        if not algorithms_to_test:
            algorithms_to_test = list(AVAILABLE_ALGORITHMS.keys())

        print(f"Starting evaluation for {len(left_image_files)} image pairs and algorithms: {algorithms_to_test}")

        all_results = []

        for i, left_img_path in enumerate(left_image_files):
            base_name_left = os.path.basename(left_img_path)
            image_pair_id = base_name_left.replace("left_", "").replace(".jpg", "")
            right_img_path = os.path.join(data_dir, "right", base_name_left.replace("left_", "right_"))
            annotation_path = os.path.join(annotations_dir, f"{image_pair_id}_annotation.json")

            print(f"\nProcessing pair {i+1}/{len(left_image_files)}: {image_pair_id}")

            if not os.path.exists(right_img_path):
                print(f"  Right image not found: {right_img_path}. Skipping pair.")
                continue
            
            gt_data = load_ground_truth_annotation(annotation_path)
            if not gt_data or "avg_disparity" not in gt_data:
                print(f"  Ground truth not found or invalid for {image_pair_id} at {annotation_path}. Skipping pair.")
                continue
            
            ground_truth_disp = gt_data["avg_disparity"]

            for algo_name in algorithms_to_test:
                print(f"  Testing with algorithm: {algo_name}")
                MatcherClass = AVAILABLE_ALGORITHMS.get(algo_name)
                if not MatcherClass:
                    print(f"    Algorithm '{algo_name}' not found. Skipping.")
                    continue
                
                matcher_instance = MatcherClass() # Initialize with default params
                
                # TODO: Implement parameter tuning loop here if desired,
                # or assume default params for now.
                # For the proposal, you'd iterate over tunable parameter sets.
                # For simplicity in this first pass, using defaults.

                pair_result = self.evaluate_pair(left_img_path, right_img_path, ground_truth_disp,
                                                 matcher_instance, debug_display=debug_each_step)
                
                if pair_result == "quit":
                    print("Quitting evaluation early by user request.")
                    if self.yolo_detector: self.yolo_detector.stop()
                    cv2.destroyAllWindows()
                    return pd.DataFrame(all_results) if all_results else None

                if pair_result:
                    all_results.append(pair_result)
                
                if debug_each_step: cv2.destroyAllWindows() # Clean up windows from evaluate_pair

        if self.yolo_detector:
            self.yolo_detector.stop()
        
        cv2.destroyAllWindows() # Final cleanup

        if not all_results:
            print("No results generated.")
            return None

        results_df = pd.DataFrame(all_results)
        
        # Calculate overall metrics per algorithm as per proposal
        # - Average absolute disparity error (pixels)
        # - Standard deviation of disparity error (pixels)
        # - Average runtime (milliseconds per frame)
        summary_stats = results_df.groupby('algorithm').agg(
            avg_abs_error_roi=('absolute_error_roi', 'mean'),
            std_dev_abs_error_roi=('absolute_error_roi', 'std'),
            avg_runtime_ms=('runtime_ms', 'mean'),
            num_pairs_processed=('absolute_error_roi', 'count') # Count non-NaN errors
        ).reset_index()

        print("\n--- Evaluation Summary ---")
        print(summary_stats.to_string())

        # Save results
        results_df.to_csv(os.path.join(output_dir, "detailed_evaluation_results.csv"), index=False)
        summary_stats.to_csv(os.path.join(output_dir, "summary_evaluation_stats.csv"), index=False)
        print(f"\nDetailed results saved to {os.path.join(output_dir, 'detailed_evaluation_results.csv')}")
        print(f"Summary stats saved to {os.path.join(output_dir, 'summary_evaluation_stats.csv')}")
        
        return results_df, summary_stats

    def __del__(self):
        if self.yolo_detector:
            self.yolo_detector.stop()