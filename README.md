# Real-Time Basketball Distance Estimation - Stereo Vision Algorithm Evaluation

This project implements the evaluation framework described in the proposal "Real-Time Basketball Distance Estimation with AI-Assisted Stereo Vision" by Sai Bhakta (CS117, 23 May 2025).

The primary goal is to evaluate a range of stereo-matching techniques and identify the approach that delivers the highest disparity accuracy with minimal computational overhead for a basketball rebounding robot.

## Project Structure
stereo_evaluation_project/
├── data/ # Data storage
│ ├── raw_stereo_pairs/ # Raw synchronized images from cameras
│ │ ├── left/
│ │ └── right/
│ ├── calibration_images/ # Checkerboard images for calibration
│ │ ├── left/
│ │ └── right/
│ ├── calibrated_params/ # Stores stereo_calibration.npz
│ └── ground_truth_annotations/ # Stores annotated disparity data (e.g., JSON/CSV files)
├── resources/ # External resources like AI models
│ └── yolov11n.hef # YOLOv11-n model (ensure this path is correct)
├── src/ # Source code
│ ├── disparity_algorithms/ # Implementations of stereo matching algorithms
│ │ ├── init.py
│ │ ├── base_disparity.py # Abstract base class for matchers
│ │ ├── block_matching.py # Stereo Block Matching
│ │ ├── sgbm.py # Semi-Global Block Matching
│ │ ├── census_cost_matching.py # Census-Cost Matching (placeholder)
│ │ ├── cascade_matching.py # Cascade Matching (placeholder)
│ │ ├── efficient_large_scale_stereo.py # Efficient Large-Scale Stereo (placeholder)
│ │ └── feature_based_matching.py # Feature-Based Matching (placeholder)
│ ├── init.py
│ ├── calibration.py # Stereo camera calibration logic
│ ├── data_acquisition.py # Script to capture stereo images
│ ├── evaluation_pipeline.py # Core evaluation logic and metrics calculation
│ ├── ground_truth_tool.py # Tool for manual disparity annotation
│ ├── main_evaluator.py # Main script to run the evaluation
│ ├── utils.py # Utility functions (image loading, processing)
│ └── yolo_detector.py # YOLOv11-n inference for basketball detection
├── README.md # This file
└── requirements.txt # Python dependencies
## Setup

1.  **Clone the repository (or create the structure).**
2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Hailo AI SDK:** This project uses a GStreamer pipeline with Hailo AI components for YOLOv11-n inference. Ensure you have the Hailo AI SDK installed and your environment configured correctly.
    *   Follow Hailo's official documentation for installation.
    *   Make sure the `TAPPAS_POST_PROC_DIR` environment variable is set if required by your Hailo setup.
    *   Place the `yolov11n.hef` model file in the `resources/` directory (or update the path in `src/yolo_detector.py` if necessary). The default path in `pipeline_helpers.py` from your original codebase is `/home/sai/Robot/resources/yolov11n.hef`. You might need to adjust this in a config or directly in `yolo_detector.py` for this standalone project.

4.  **Hardware (for data acquisition & calibration):**
    *   Two Raspberry Pi Camera Module 3 units, connected to your Raspberry Pi.

## Workflow

The evaluation process follows these main steps:

### 1. Camera Calibration

Calibrate your stereo camera setup to obtain intrinsic and extrinsic parameters.

*   **Capture Calibration Images:**
    *   Use a checkerboard pattern.
    *   Modify `src/data_acquisition.py` or use a similar script (like your original `camera_capture_pics.py`) to capture multiple pairs of checkerboard images from both cameras. Store them in `data/calibration_images/left/` and `data/calibration_images/right/`.
    *   Ensure filenames are synchronized (e.g., `left_01.jpg`, `right_01.jpg`).
*   **Run Calibration:**
    ```bash
    python src/calibration.py --images_dir data/calibration_images --output_file data/calibrated_params/stereo_calibration.npz
    ```
    *   Adjust `CHECKERBOARD_SIZE` and `SQUARE_SIZE` in `src/calibration.py` if needed.
*   **Sanity Check:**
    The calibration script can optionally display undistorted images. Visually inspect them to ensure the calibration was successful.

### 2. Data Acquisition (Basketball in Flight)

Capture synchronized stereo image pairs of a basketball.

*   Run the `src/data_acquisition.py` script (or adapt your existing `camera_capture_pics.py`).
    ```bash
    python src/data_acquisition.py --output_dir data/raw_stereo_pairs
    ```
*   This will save images to `data/raw_stereo_pairs/left/` and `data/raw_stereo_pairs/right/`.
*   Collect a diverse set of images under varied lighting conditions (indoor/outdoor) as per your proposal.

### 3. Ground Truth Annotation

Manually annotate disparity for the basketball in your captured image pairs.

*   Run the `src/ground_truth_tool.py`:
    ```bash
    python src/ground_truth_tool.py --images_dir data/raw_stereo_pairs --calibration_file data/calibrated_params/stereo_calibration.npz --annotations_dir data/ground_truth_annotations
    ```
*   This tool will:
    1.  Load a rectified stereo pair.
    2.  Allow you to click on corresponding points on the basketball in the left and right images.
    3.  It will calculate and save the average disparity for the ball for that image pair.
    4.  Annotations will be saved (e.g., as JSON files) in `data/ground_truth_annotations/`.

### 4. Run Evaluation

Execute the main evaluation script to test the implemented stereo-matching algorithms.

*   Ensure you have:
    *   Calibrated parameters (`data/calibrated_params/stereo_calibration.npz`).
    *   Raw stereo pairs (`data/raw_stereo_pairs/`).
    *   Ground truth annotations (`data/ground_truth_annotations/`).
    *   YOLO model (`resources/yolov11n.hef`).
*   Run the evaluator:
    ```bash
    python src/main_evaluator.py \
        --data_dir data/raw_stereo_pairs \
        --calibration_file data/calibrated_params/stereo_calibration.npz \
        --annotations_dir data/ground_truth_annotations \
        --results_dir results
    ```
*   The script will:
    1.  Load image pairs and their ground truth.
    2.  Perform undistortion and rectification.
    3.  Use `YoloDetector` to find the basketball ROI.
    4.  Crop images to the ROI.
    5.  For each implemented stereo-matching algorithm:
        *   Compute the disparity map.
        *   Measure execution time.
        *   Compare the computed disparity with the ground truth.
    6.  Calculate and report metrics (average absolute disparity error, standard deviation of error, average runtime).
    7.  Save results to the `results_dir`.

## Implementing Disparity Algorithms

The `src/disparity_algorithms/` directory contains a base class `BaseDisparityMatcher` and placeholders for the algorithms listed in the proposal. You will need to:

1.  Implement the `compute_disparity(self, left_roi_image, right_roi_image)` method for each algorithm.
2.  Implement `get_params(self)` and `set_params(self, **params)` to handle tunable parameters.
3.  Refer to OpenCV documentation (`cv2.StereoBM_create`, `cv2.StereoSGBM_create`, etc.) and relevant research papers for implementations.

## Verification and Debugging

The proposal outlines a verification strategy:

1.  **Calibration Sanity Check:** Visually inspect undistorted images (can be part of `src/calibration.py` or `src/utils.py`).
2.  **Test Data Debugging:**
    *   The `evaluation_pipeline.py` or `utils.py` should include options to display the YOLO-detected ROI to ensure accurate cropping.
    *   It should also allow superimposing manually annotated points or epipolar lines on images to verify ground-truth consistency.

Testing Workflow:
1. Run pipeline_app.py to Process Raw Images and Generate Cropped Data + ROI Metadata:
Open your terminal in the root directory of the project (Disparity/) and run:
Apply to README.md
Run
pipeline_app
This will:
Load raw images from data/basketball_images/.
Rectify them using the calibration file.
Run YOLO to detect objects (e.g., basketballs).
Save the rectified, cropped images of the detected objects to data/cropped_images/left/ and data/cropped_images/right/.
Generate data/cropped_images/roi_metadata.json containing metadata about each crop, including the crop_bbox_in_rectified_image and the crucial detection_bbox_in_rectified_image.
Check app.log for any errors or information.
Let it run until it processes all images and prints "All images processed by picamera_thread, sending EOS to appsrc." and then gracefully shuts down.
2. Run disparity_processor.py with the ROIDisparity Algorithm:
Once pipeline_app.py has finished, run the disparity processor:
Apply to README.md
Run
disparity_data
This will:
Load the roi_metadata.json.
Load the cropped image pairs.
For each pair, use the ROIDisparity algorithm. This algorithm will use the detection_bbox_in_rectified_image from the metadata to calculate disparity (center_x_left_detection - center_x_right_detection).
Save the results (including computed disparity and runtime) to results/disparity_data/disparity_results_ROIDisparity.csv.
Check the console output and logs for any warnings (e.g., if metadata is missing for some items).
3. Run results_analyzer.py to Compare with Ground Truth:
Finally, analyze the results produced by ROIDisparity against your ground truth data:
Apply to README.md
Run
analysis
This will:
Load disparity_results_ROIDisparity.csv.
Load the ground truth JSON files from data/annotations/ based on image_pair_id.
Calculate the absolute error between the computed_disparity from ROIDisparity and the gt_avg_disparity.
Save a detailed CSV with these comparisons to results/analysis/detailed_analysis_results_ROIDisparity.csv.
Save and print a summary CSV with overall metrics (average error, std dev error, average runtime, etc.) to results/analysis/summary_analysis_stats_ROIDisparity.csv.
Expected Outcomes & Verification:
data/cropped_images/: Should contain left/ and right/ subdirectories with rectified, cropped images.
data/cropped_images/roi_metadata.json: Should exist and be a valid JSON containing entries for each cropped image, with both crop_bbox_in_rectified_image and detection_bbox_in_rectified_image fields.
results/disparity_data/disparity_results_ROIDisparity.csv: Should contain columns like image_pair_id, computed_disparity, runtime_ms, etc. The computed_disparity values should be the result of (left_detection_center_x - right_detection_center_x).
results/analysis/detailed_analysis_results_ROIDisparity.csv: Will have the data from the previous CSV plus gt_avg_disparity and absolute_error.
results/analysis/summary_analysis_stats_ROIDisparity.csv (and console output): Will show the performance metrics for the ROIDisparity algorithm. Since ROIDisparity is very basic, the error might be significant, but the pipeline itself should function.
This provides a complete end-to-end test of the new refactored system using your newly created ROIDisparity algorithm. Remember to replace placeholder paths/parameters with your actual ones if they differ.
