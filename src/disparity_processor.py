import os
import json
import cv2
import numpy as np
import pandas as pd
import time
import argparse
import logging

# Assuming disparity_algorithms and utils are structured to be importable
# If src is a package, and these are modules within src:
from .disparity_algorithms import AVAILABLE_ALGORITHMS, BaseDisparityMatcher
from .utils import load_image # Assuming load_image can handle color images by default

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DisparityProcessor:
    def __init__(self, roi_metadata_path: str, cropped_images_dir: str, 
                 algorithm_name: str, output_dir: str):
        self.roi_metadata_path = roi_metadata_path
        self.cropped_images_dir = cropped_images_dir
        self.algorithm_name = algorithm_name
        self.output_dir = output_dir

        if not os.path.exists(self.roi_metadata_path):
            logging.error(f"ROI metadata file not found: {self.roi_metadata_path}")
            raise FileNotFoundError(f"ROI metadata file not found: {self.roi_metadata_path}")
        
        with open(self.roi_metadata_path, 'r') as f:
            self.roi_metadata = json.load(f)
        logging.info(f"Successfully loaded ROI metadata from {self.roi_metadata_path}")

        if self.algorithm_name not in AVAILABLE_ALGORITHMS:
            logging.error(f"Algorithm '{self.algorithm_name}' not found. Available: {list(AVAILABLE_ALGORITHMS.keys())}")
            raise ValueError(f"Algorithm '{self.algorithm_name}' not found.")
        
        self.AlgorithmClass = AVAILABLE_ALGORITHMS[self.algorithm_name]
        logging.info(f"Selected algorithm: {self.algorithm_name}")

        os.makedirs(self.output_dir, exist_ok=True)

    def _extract_representative_disparity(self, disparity_map: np.ndarray) -> float | None:
        """
        Extracts a representative disparity value from the computed map.
        Takes the median of valid disparities in a central 50%x50% patch.
        """
        if disparity_map is None or disparity_map.size == 0:
            logging.warning("Disparity map is empty or None.")
            return None

        h_roi, w_roi = disparity_map.shape[:2]
        if h_roi == 0 or w_roi == 0:
            logging.warning("Disparity map has zero height or width.")
            return None

        center_y_start, center_y_end = int(h_roi * 0.25), int(h_roi * 0.75)
        center_x_start, center_x_end = int(w_roi * 0.25), int(w_roi * 0.75)
        
        central_patch_disp = disparity_map[center_y_start:center_y_end, center_x_start:center_x_end]
        
        # Filter out invalid disparity values (e.g., <=0, NaN, or sometimes large placeholder values)
        # Some algorithms might return 0 for no disparity, some might use negative for invalid.
        # We consider > 0 as valid for typical disparity maps.
        valid_disparities = central_patch_disp[central_patch_disp > 0] 

        if valid_disparities.size > 0:
            # Median is often more robust to outliers than mean
            return float(np.median(valid_disparities))
        else:
            logging.warning("No valid disparities found in the central patch of the computed map.")
            return None # Or 0.0, depending on how failures should be represented

    def process_disparities(self):
        results_list = []
        processed_pairs = 0

        # Iterate through metadata, assuming keys are cropped filenames
        for left_cropped_filename, left_metadata_item in self.roi_metadata.items():
            if left_metadata_item.get('side') != 'left':
                continue # Process pairs starting from the left image

            image_pair_id = left_metadata_item.get('image_pair_id', 'unknown_pair')
            logging.info(f"Processing pair ID: {image_pair_id}, left cropped: {left_cropped_filename}")

            # Construct right cropped filename based on left one
            # Assuming naming convention like "left_XYZ_cropped.jpg" -> "right_XYZ_cropped.jpg"
            if "_cropped" in left_cropped_filename and left_cropped_filename.startswith("left_"):
                 base_name_part = left_cropped_filename[len("left_") : left_cropped_filename.rfind("_cropped")]
                 right_cropped_filename = f"right_{base_name_part}_cropped{os.path.splitext(left_cropped_filename)[1]}"
            else:
                logging.warning(f"Could not determine right cropped filename for {left_cropped_filename}. Skipping.")
                continue
            
            right_metadata_item = self.roi_metadata.get(right_cropped_filename)
            if not right_metadata_item:
                logging.warning(f"Right image metadata not found for {right_cropped_filename} (corresponding to {left_cropped_filename}). Skipping pair.")
                continue

            left_cropped_path = os.path.join(self.cropped_images_dir, 'left', left_cropped_filename)
            right_cropped_path = os.path.join(self.cropped_images_dir, 'right', right_cropped_filename)

            if not os.path.exists(left_cropped_path):
                logging.warning(f"Left cropped image not found: {left_cropped_path}. Skipping.")
                continue
            if not os.path.exists(right_cropped_path):
                logging.warning(f"Right cropped image not found: {right_cropped_path}. Skipping.")
                continue
            
            try:
                # Load images. BaseDisparityMatcher implementations should handle color/grayscale.
                img_left = load_image(left_cropped_path, grayscale=False) # Load as color by default
                img_right = load_image(right_cropped_path, grayscale=False)
            except FileNotFoundError as e:
                logging.error(f"Error loading images: {e}. Skipping pair.")
                continue
            except Exception as e: # Catch other potential cv2 errors
                logging.error(f"A general error occurred loading images {left_cropped_path} or {right_cropped_path}: {e}. Skipping pair.")
                continue


            if img_left.size == 0 or img_right.size == 0:
                 logging.warning(f"Loaded empty image for {left_cropped_filename} or {right_cropped_filename}. Skipping.")
                 continue


            matcher_instance = self.AlgorithmClass()
            
            start_time = time.perf_counter()
            # Pass the full metadata items to the algorithm
            computed_disparity_map = matcher_instance.compute_disparity(img_left, img_right, 
                                                                      left_metadata_item, right_metadata_item)
            runtime_ms = (time.perf_counter() - start_time) * 1000

            if computed_disparity_map is None or computed_disparity_map.size == 0: # Check for empty array too
                logging.warning(f"Algorithm {self.algorithm_name} failed to compute disparity or returned empty map for {left_cropped_filename}. Skipping.")
                computed_disparity_local = None # Ensure it's None if map is bad
            else:
                computed_disparity_local = self._extract_representative_disparity(computed_disparity_map)

            # if computed_disparity_local is None: # This check is now implicitly handled by assigning None above or from extract method
            #     logging.warning(f"Could not extract representative disparity for {left_cropped_filename} using {self.algorithm_name}. Skipping.")
            
            # Disparity is already "global" in the sense that it's the disparity of the object
            # within the rectified coordinate system. The ROI defines *where* this measurement was made.
            
            # Retrieve crop bbox info for logging, using .get for safety
            crop_bbox_left = left_metadata_item.get('crop_bbox_in_rectified_image', {})

            result_item = {
                "image_pair_id": image_pair_id,
                "left_cropped_image": left_cropped_filename,
                "right_cropped_image": right_cropped_filename,
                "algorithm": self.algorithm_name,
                "params": json.dumps(matcher_instance.get_params()),
                "runtime_ms": runtime_ms,
                "computed_disparity": computed_disparity_local, # Can be None
                "left_crop_bbox_x_rect": crop_bbox_left.get('x'),
                "left_crop_bbox_y_rect": crop_bbox_left.get('y'),
                "left_crop_bbox_w_rect": crop_bbox_left.get('w'),
                "left_crop_bbox_h_rect": crop_bbox_left.get('h'),
                "detection_confidence_left": left_metadata_item.get("detection_confidence")
            }
            results_list.append(result_item)
            processed_pairs += 1
        
        logging.info(f"Processed {processed_pairs} image pairs for algorithm {self.algorithm_name}.")

        if not results_list:
            logging.warning("No results were generated.")
            return

        results_df = pd.DataFrame(results_list)
        output_csv_path = os.path.join(self.output_dir, f"disparity_results_{self.algorithm_name}.csv")
        
        try:
            results_df.to_csv(output_csv_path, index=False)
            logging.info(f"Disparity processing results saved to {output_csv_path}")
        except Exception as e:
            logging.error(f"Failed to save results to CSV: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process cropped stereo images to compute disparities using a specified algorithm.")
    parser.add_argument("--roi_metadata_file", type=str, required=True,
                        help="Path to the roi_metadata.json file generated by pipeline_app.py.")
    parser.add_argument("--cropped_images_dir", type=str, required=True,
                        help="Base directory where cropped_images/left and cropped_images/right are stored.")
    parser.add_argument("--algorithm", type=str, required=True,
                        help=f"Name of the disparity algorithm to use. Available: {', '.join(AVAILABLE_ALGORITHMS.keys())}")
    parser.add_argument("--output_dir", type=str, default="results/disparity_data",
                        help="Directory to save the output CSV file with disparity results.")
    
    args = parser.parse_args()

    try:
        processor = DisparityProcessor(
            roi_metadata_path=args.roi_metadata_file,
            cropped_images_dir=args.cropped_images_dir,
            algorithm_name=args.algorithm,
            output_dir=args.output_dir
        )
        processor.process_disparities()
    except Exception as e:
        logging.error(f"An error occurred during disparity processing: {e}", exc_info=True)

if __name__ == "__main__":
    main() 