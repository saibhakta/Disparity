import argparse
import os
from evaluation_pipeline import StereoEvaluator
from disparity_algorithms import AVAILABLE_ALGORITHMS # To list available ones

def main():
    parser = argparse.ArgumentParser(description="Main script to run stereo disparity algorithm evaluation.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing 'left' and 'right' subdirs with raw stereo pairs.")
    parser.add_argument("--calibration_file", type=str, required=True,
                        help="Path to the stereo_calibration.npz file.")
    parser.add_argument("--annotations_dir", type=str, required=True,
                        help="Directory containing ground truth JSON annotation files.")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to save evaluation results (CSVs, plots).")
    
    # YOLO specific paths - make them optional if YOLO is not strictly needed for a run
    parser.add_argument("--yolo_hef_path", type=str, default="resources/yolov11n.hef",
                        help="Path to the YOLO .hef model file. If not provided, ROI detection might be skipped or use full image.")
    parser.add_argument("--yolo_so_path", type=str,
                        default=os.path.join(os.environ.get('TAPPAS_POST_PROC_DIR', '/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/'), 'libyolo_hailortpp_postprocess.so'), # Default based on common Hailo setup
                        help="Path to the YOLO postprocessing .so file for Hailo GStreamer. Critical for YOLO functionality.")

    parser.add_argument("--algorithms", nargs='+', type=str, default=None, # if None, will run all available
                        help=f"Space-separated list of algorithms to test. Available: {', '.join(AVAILABLE_ALGORITHMS.keys())}. Default: all.")
    parser.add_argument("--debug_steps", action='store_true',
                        help="Show intermediate images and results for each step during evaluation for debugging.")

    args = parser.parse_args()

    # Validate paths
    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found.")
        return
    if not os.path.isfile(args.calibration_file):
        print(f"Error: Calibration file '{args.calibration_file}' not found.")
        return
    if not os.path.isdir(args.annotations_dir):
        print(f"Error: Annotations directory '{args.annotations_dir}' not found.")
        return
    
    if args.yolo_hef_path and not os.path.isfile(args.yolo_hef_path):
        print(f"Warning: YOLO HEF file '{args.yolo_hef_path}' not found. YOLO ROI detection might fail or be skipped.")
        # Allow continuation if user wants to run without YOLO, assuming evaluator handles it
    
    if args.yolo_so_path and not os.path.isfile(args.yolo_so_path):
        print(f"Warning: YOLO SO file '{args.yolo_so_path}' not found. YOLO functionality might be impaired.")
        # Allow continuation, assuming evaluator can handle this or it's not critical for all algorithms

    # Check if specified algorithms are valid
    algorithms_to_run = args.algorithms
    if algorithms_to_run:
        for algo_name in algorithms_to_run:
            if algo_name not in AVAILABLE_ALGORITHMS:
                print(f"Error: Algorithm '{algo_name}' is not recognized. Available are: {', '.join(AVAILABLE_ALGORITHMS.keys())}")
                return
    else: # If --algorithms is not specified, run all
        algorithms_to_run = list(AVAILABLE_ALGORITHMS.keys())
        print(f"No specific algorithms requested. Running all available: {', '.join(algorithms_to_run)}")


    print("Initializing Stereo Evaluator...")
    evaluator = StereoEvaluator(
        calibration_file_path=args.calibration_file,
        yolo_hef_path=args.yolo_hef_path,
        yolo_so_path=args.yolo_so_path
    )

    print("Starting evaluation run...")
    evaluator.run_evaluation(
        data_dir=args.data_dir,
        annotations_dir=args.annotations_dir,
        output_dir=args.results_dir,
        algorithms_to_test=algorithms_to_run,
        debug_each_step=args.debug_steps
    )

    print("Evaluation finished.")

if __name__ == "__main__":
    main()