import os
import json
import pandas as pd
import numpy as np
import argparse
import logging

# Assuming utils.py is structured to be importable
# If src is a package, and utils is a module within src:
from utils import load_ground_truth_annotation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ResultsAnalyzer:
    def __init__(self, disparity_results_csv_path: str, annotations_dir: str, output_dir: str):
        self.disparity_results_csv_path = disparity_results_csv_path
        self.annotations_dir = annotations_dir
        self.output_dir = output_dir

        if not os.path.exists(self.disparity_results_csv_path):
            logging.error(f"Disparity results CSV not found: {self.disparity_results_csv_path}")
            raise FileNotFoundError(f"Disparity results CSV not found: {self.disparity_results_csv_path}")
        
        try:
            self.results_df = pd.read_csv(self.disparity_results_csv_path)
            logging.info(f"Successfully loaded disparity results from {self.disparity_results_csv_path}")
        except Exception as e:
            logging.error(f"Error loading disparity results CSV: {e}")
            raise

        if not os.path.isdir(self.annotations_dir):
            logging.error(f"Annotations directory not found: {self.annotations_dir}")
            raise FileNotFoundError(f"Annotations directory not found: {self.annotations_dir}")

        os.makedirs(self.output_dir, exist_ok=True)
        
        # Extract algorithm name from the CSV filename if possible for output naming
        self.algorithm_name = "unknown_algorithm"
        base_csv_name = os.path.basename(self.disparity_results_csv_path)
        if base_csv_name.startswith("disparity_results_") and base_csv_name.endswith(".csv"):
            self.algorithm_name = base_csv_name[len("disparity_results_"):-len(".csv")]

    def analyze_results(self):
        if self.results_df.empty:
            logging.warning("Disparity results DataFrame is empty. No analysis to perform.")
            return

        self.results_df['gt_avg_disparity'] = np.nan
        self.results_df['absolute_error'] = np.nan

        successfully_compared_pairs = 0

        for index, row in self.results_df.iterrows():
            image_pair_id = str(row['image_pair_id']) # Ensure it's a string for filename matching
            computed_disp = row['computed_disparity']

            # Attempt to find the ground truth annotation file.
            # Common naming conventions: {id}.json, {id}_annotation.json
            # The example was 4.json for image_pair_name "4"
            potential_gt_filenames = [
                f"{image_pair_id}.json",
                f"{image_pair_id}_annotation.json", 
                f"{image_pair_id.lstrip('0')}.json", # e.g. if id is "04", try "4.json"
                f"{image_pair_id.lstrip('0')}_annotation.json"
            ]
            
            gt_annotation_path = None
            for fname in potential_gt_filenames:
                current_path = os.path.join(self.annotations_dir, fname)
                if os.path.exists(current_path):
                    gt_annotation_path = current_path
                    break
            
            if not gt_annotation_path:
                logging.warning(f"Ground truth annotation file not found for image_pair_id '{image_pair_id}' in {self.annotations_dir}. Tried: {potential_gt_filenames}")
                continue

            gt_data = load_ground_truth_annotation(gt_annotation_path)

            if gt_data and 'avg_disparity' in gt_data:
                gt_disp = gt_data['avg_disparity']
                self.results_df.loc[index, 'gt_avg_disparity'] = gt_disp
                
                if pd.notna(computed_disp):
                    abs_error = abs(computed_disp - gt_disp)
                    self.results_df.loc[index, 'absolute_error'] = abs_error
                    successfully_compared_pairs +=1
                else:
                    logging.warning(f"Computed disparity is NaN/None for image_pair_id '{image_pair_id}'. Cannot compute error.")
            else:
                logging.warning(f"Ground truth data for '{image_pair_id}' (path: {gt_annotation_path}) is missing 'avg_disparity' or is invalid.")

        logging.info(f"Successfully compared {successfully_compared_pairs} pairs against ground truth.")

        # Calculate summary statistics
        # Ensure calculations are on rows where absolute_error is not NaN
        valid_errors_df = self.results_df.dropna(subset=['absolute_error'])

        if not valid_errors_df.empty:
            avg_abs_error = valid_errors_df['absolute_error'].mean()
            std_dev_abs_error = valid_errors_df['absolute_error'].std()
        else:
            avg_abs_error = np.nan
            std_dev_abs_error = np.nan
            logging.warning("No valid absolute errors to calculate overall mean/std_dev.")

        # Runtime stats can be calculated on all processed entries, even if GT was missing or computed_disp was None
        avg_runtime_ms = self.results_df['runtime_ms'].mean()
        std_dev_runtime_ms = self.results_df['runtime_ms'].std()
        num_total_processed_by_algo = len(self.results_df)
        num_with_computed_disp = self.results_df['computed_disparity'].count() # Non-NaN computed disparities

        summary_stats_data = {
            'algorithm': [self.algorithm_name],
            'avg_absolute_error_pixels': [avg_abs_error],
            'std_dev_absolute_error_pixels': [std_dev_abs_error],
            'avg_runtime_ms_per_pair': [avg_runtime_ms],
            'std_dev_runtime_ms_per_pair': [std_dev_runtime_ms],
            'num_pairs_processed_by_algorithm': [num_total_processed_by_algo],
            'num_pairs_with_computed_disparity': [num_with_computed_disp],
            'num_pairs_successfully_compared_with_gt': [successfully_compared_pairs]
        }
        summary_df = pd.DataFrame(summary_stats_data)

        # Save detailed results (with GT and error columns)
        detailed_output_csv_path = os.path.join(self.output_dir, f"detailed_analysis_results_{self.algorithm_name}.csv")
        try:
            self.results_df.to_csv(detailed_output_csv_path, index=False)
            logging.info(f"Detailed analysis results saved to {detailed_output_csv_path}")
        except Exception as e:
            logging.error(f"Failed to save detailed analysis results: {e}")

        # Save summary statistics
        summary_output_csv_path = os.path.join(self.output_dir, f"summary_analysis_stats_{self.algorithm_name}.csv")
        try:
            summary_df.to_csv(summary_output_csv_path, index=False)
            logging.info(f"Summary analysis stats saved to {summary_output_csv_path}")
        except Exception as e:
            logging.error(f"Failed to save summary analysis stats: {e}")
        
        print("\n--- Analysis Summary ---")
        try:
            print(summary_df.to_string(index=False))
        except Exception as e:
            logging.error(f"Error printing summary_df to string: {e}")
            print(summary_df) # Fallback to default print

def main():
    parser = argparse.ArgumentParser(description="Analyzes disparity processing results against ground truth annotations.")
    parser.add_argument("--disparity_results_csv", type=str, required=True,
                        help="Path to the CSV file generated by disparity_processor.py (e.g., disparity_results_ALGO.csv)")
    parser.add_argument("--annotations_dir", type=str, required=True,
                        help="Directory containing ground truth JSON annotation files (e.g., data/annotations/).")
    parser.add_argument("--output_dir", type=str, default="results/analysis",
                        help="Directory to save the output analysis CSV files.")
    
    args = parser.parse_args()

    try:
        analyzer = ResultsAnalyzer(
            disparity_results_csv_path=args.disparity_results_csv,
            annotations_dir=args.annotations_dir,
            output_dir=args.output_dir
        )
        analyzer.analyze_results()
    except Exception as e:
        logging.error(f"An error occurred during results analysis: {e}", exc_info=True)

if __name__ == "__main__":
    main() 