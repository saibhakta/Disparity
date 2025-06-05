import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import multiprocessing
import numpy as np
import setproctitle
import cv2
import time
from picamera2 import Picamera2
import signal
from gstreamer_helpers import disable_qos
import threading
from datetime import datetime
import logging
import glob
import hailo
from gstreamer_helpers import get_numpy_from_buffer
import json
from utils import CalibrationData

# Global map to associate PTS with original file paths
pts_to_filepath_map = {}
OUTPUT_CROPPED_BASE_DIR = "data/cropped_images"
# Define a placeholder path for the calibration file.
# THIS SHOULD BE CONFIGURABLE OR PASSED AS AN ARGUMENT IN A REAL APPLICATION.
CALIBRATION_FILE_PATH = "data/calibration_data/stereo_calibration.npz"
ROI_METADATA_FILENAME = "roi_metadata.json"
global_roi_metadata_dict = {} # Dictionary to store ROI metadata, keyed by cropped_filename

def PIPELINE_STRING(display=True):
    if display:
        return "appsrc name=app_source is-live=true leaky-type=downstream max-buffers=1 ! video/x-raw, format=RGB, width=2304, height=1296 !  queue name=source_scale_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoscale name=source_videoscale n-threads=2 ! queue name=source_convert_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoconvert n-threads=3 name=source_convert qos=false ! video/x-raw, pixel-aspect-ratio=1/1, format=RGB, width=2304, height=1296  ! queue name=inference_wrapper_input_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailocropper name=inference_wrapper_crop so-path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true hailoaggregator name=inference_wrapper_agg inference_wrapper_crop. ! queue name=inference_wrapper_bypass_q leaky=downstream max-size-buffers=10 max-size-bytes=0 max-size-time=0  ! inference_wrapper_agg.sink_0 inference_wrapper_crop. ! queue name=inference_scale_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoscale name=inference_videoscale n-threads=2 qos=false ! queue name=inference_convert_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! video/x-raw, pixel-aspect-ratio=1/1 ! videoconvert name=inference_videoconvert n-threads=2 ! queue name=inference_hailonet_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailonet name=inference_hailonet hef-path=/home/sai/Robot/resources/yolov11n.hef batch-size=2  vdevice-group-id=1 nms-score-threshold=0.3 nms-iou-threshold=0.45 output-format-type=HAILO_FORMAT_TYPE_FLOAT32 force-writable=true  ! queue name=inference_hailofilter_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailofilter name=inference_hailofilter so-path=/home/sai/Robot/venv_hailo_rpi5_examples/lib/python3.11/site-packages/hailo_apps_infra/../resources/libyolo_hailortpp_postprocess.so   function-name=filter_letterbox  qos=false ! queue name=inference_output_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0   ! inference_wrapper_agg.sink_1 inference_wrapper_agg. ! queue name=inference_wrapper_output_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0   ! queue name=identity_callback_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! identity name=identity_callback ! queue name=hailo_display_overlay_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailooverlay name=hailo_display_overlay  ! queue name=hailo_display_videoconvert_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoconvert name=hailo_display_videoconvert n-threads=2 qos=false ! queue name=hailo_display_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! fpsdisplaysink name=hailo_display video-sink=autovideosink sync=false text-overlay=False signal-fps-measurements=true"
    
    # No display
    return "appsrc name=app_source is-live=true leaky-type=upstream max-buffers=1 ! video/x-raw, format=RGB, width=2304, height=1296 !  queue name=source_scale_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoscale name=source_videoscale n-threads=2 ! queue name=source_convert_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoconvert n-threads=3 name=source_convert qos=false ! video/x-raw, pixel-aspect-ratio=1/1, format=RGB, width=2304, height=1296  ! queue name=inference_wrapper_input_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailocropper name=inference_wrapper_crop so-path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true hailoaggregator name=inference_wrapper_agg inference_wrapper_crop. ! queue name=inference_wrapper_bypass_q leaky=downstream max-size-buffers=10 max-size-bytes=0 max-size-time=0  ! inference_wrapper_agg.sink_0 inference_wrapper_crop. ! queue name=inference_scale_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoscale name=inference_videoscale n-threads=2 qos=false ! queue name=inference_convert_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! video/x-raw, pixel-aspect-ratio=1/1 ! videoconvert name=inference_videoconvert n-threads=2 ! queue name=inference_hailonet_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailonet name=inference_hailonet hef-path=/home/sai/Robot/resources/yolov11n.hef batch-size=2  vdevice-group-id=1 nms-score-threshold=0.3 nms-iou-threshold=0.45 output-format-type=HAILO_FORMAT_TYPE_FLOAT32 force-writable=true  ! queue name=inference_hailofilter_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailofilter name=inference_hailofilter so-path=/home/sai/Robot/venv_hailo_rpi5_examples/lib/python3.11/site-packages/hailo_apps_infra/../resources/libyolo_hailortpp_postprocess.so   function-name=filter_letterbox  qos=false ! queue name=inference_output_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0   ! inference_wrapper_agg.sink_1 inference_wrapper_agg. ! queue name=inference_wrapper_output_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0   ! queue name=identity_callback_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! identity name=identity_callback ! fakesink"


def camera_callback(pad, info):
    buffer: Gst.Buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    original_filepath = pts_to_filepath_map.get(buffer.pts)
    # if original_filepath is None: # This check is a bit redundant given the logging below
        # logging.warning(f"PTS {buffer.pts} not found in filepath map. Cannot save cropped image or metadata.")
        # return Gst.PadProbeReturn.OK # Potentially return early if no path
    
    output_subdir_name = None
    side_flag_value = 0 # 0 for unknown, 1 for left, 2 for right
    if buffer.has_flags(Gst.BufferFlags(1)): # LEFT
        output_subdir_name = "left"
        side_flag_value = 1
    elif buffer.has_flags(Gst.BufferFlags(2)): # RIGHT
        output_subdir_name = "right"
        side_flag_value = 2
    # else: # This logging is now handled better below with original_filepath check
        # logging.error("Buffer has no side flag. Cannot determine output folder for cropped image.")
        # return Gst.PadProbeReturn.OK

    pad_caps = pad.get_current_caps()
    if not pad_caps:
        logging.error("Failed to get caps from pad.")
        return Gst.PadProbeReturn.OK
    
    s = pad_caps.get_structure(0)
    if not s:
        logging.error("Failed to get structure from caps.")
        return Gst.PadProbeReturn.OK

    res, frame_width = s.get_int("width")
    if not res:
        logging.error("Failed to get width from caps.")
        return Gst.PadProbeReturn.OK 
    res, frame_height = s.get_int("height")
    if not res:
        logging.error("Failed to get height from caps.")
        return Gst.PadProbeReturn.OK 
    
    bgr_frame = get_numpy_from_buffer(buffer, s.get_string("format") if s.has_field("format") else "RGB", frame_width, frame_height)

    if bgr_frame is None:
        logging.error("Failed to get numpy array from buffer.")
        return Gst.PadProbeReturn.OK

    # At this point, bgr_frame is a rectified image (assuming picamera_thread sends rectified ones)

    roi = hailo.get_roi_from_buffer(buffer)
    hailo_detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    # hailo_detections: list[hailo.HAILO_DETECTION] # Type hint already present

    padding = 20  # pixels
    detections_saved_count = 0

    current_image_identifier = "unknown_image_at_pts_" + str(buffer.pts)
    if original_filepath:
        current_image_identifier = os.path.basename(original_filepath)
    elif not original_filepath:
        logging.warning(f"PTS {buffer.pts} not found in filepath map. Essential metadata for cropped image and ROI will be missing or incomplete.")
        # Decide if to proceed without original_filepath, impacts metadata quality.
        # For now, we'll proceed but log warnings when data is missing.

    if not output_subdir_name: # Check if side was determined
        logging.error(f"Buffer for {current_image_identifier} (PTS: {buffer.pts}) has no side flag. Cannot determine output folder or save ROI metadata correctly.")
        return Gst.PadProbeReturn.OK # Cannot proceed without side


    if not hailo_detections:
        logging.info(f"No detections found in {current_image_identifier}.")
        return Gst.PadProbeReturn.OK

    highest_confidence_detection = None
    max_confidence = -1.0

    for detection in hailo_detections:
        confidence = detection.get_confidence()
        if confidence > max_confidence:
            max_confidence = confidence
            highest_confidence_detection = detection
    
    if highest_confidence_detection:
        detection = highest_confidence_detection
        bbox = detection.get_bbox()

        # Coordinates are relative to the (rectified) bgr_frame
        # These are the coordinates of the raw detection box
        detection_xmin_abs = bbox.xmin() * frame_width
        detection_ymin_abs = bbox.ymin() * frame_height
        detection_width_abs = bbox.width() * frame_width
        detection_height_abs = bbox.height() * frame_height

        # Calculate crop coordinates with padding around the detection box
        crop_x1 = max(0, int(detection_xmin_abs - padding))
        crop_y1 = max(0, int(detection_ymin_abs - padding))
        detection_box_xmax_abs = detection_xmin_abs + detection_width_abs
        detection_box_ymax_abs = detection_ymin_abs + detection_height_abs
        crop_x2 = min(frame_width, int(detection_box_xmax_abs + padding))
        crop_y2 = min(frame_height, int(detection_box_ymax_abs + padding))

        if crop_x1 < crop_x2 and crop_y1 < crop_y2:
            cropped_bgr_image = bgr_frame[crop_y1:crop_y2, crop_x1:crop_x2]
            actual_crop_w = crop_x2 - crop_x1
            actual_crop_h = crop_y2 - crop_y1

            cropped_filename = "error_generating_filename.jpg" # Default/fallback
            image_pair_id = "unknown_pair_id"

            if original_filepath:
                base_name = os.path.basename(original_filepath)
                name_part, ext_part = os.path.splitext(base_name)
                cropped_filename = f"{name_part}_cropped{ext_part}" # Standardized naming
                # Extract image_pair_id (e.g., "04" from "left_04.jpg")
                if name_part.startswith("left_"):
                    image_pair_id = name_part[len("left_"):]
                elif name_part.startswith("right_"):
                    image_pair_id = name_part[len("right_"):]
                else: # Fallback if no known prefix
                    image_pair_id = name_part 
            else:
                # Try to create a unique enough filename if original_filepath is missing
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cropped_filename = f"{output_subdir_name}_unknown_{timestamp_str}_cropped.jpg"
                logging.warning(f"Original filepath missing for PTS {buffer.pts}, generated crop filename: {cropped_filename}")
            
            final_output_dir = os.path.join(OUTPUT_CROPPED_BASE_DIR, output_subdir_name)
            full_output_path = os.path.join(final_output_dir, cropped_filename)
            
            try:
                cv2.imwrite(full_output_path, cropped_bgr_image)
                logging.info(f"Saved highest confidence cropped image (conf: {max_confidence:.2f}) to {full_output_path}")
                detections_saved_count += 1

                # Store ROI metadata
                global global_roi_metadata_dict
                roi_info = {
                    "original_image_path": original_filepath if original_filepath else "unknown",
                    "image_pair_id": image_pair_id,
                    "side": output_subdir_name, # "left" or "right"
                    "crop_bbox_in_rectified_image": { # Renamed for clarity
                        "x": crop_x1,
                        "y": crop_y1,
                        "w": actual_crop_w, 
                        "h": actual_crop_h
                    },
                    "detection_bbox_in_rectified_image": { # Added raw detection bbox
                        "x": detection_xmin_abs,
                        "y": detection_ymin_abs,
                        "w": detection_width_abs,
                        "h": detection_height_abs
                    },
                    "rectified_image_width": frame_width,
                    "rectified_image_height": frame_height,
                    "detection_confidence": float(max_confidence)
                }
                global_roi_metadata_dict[cropped_filename] = roi_info
                logging.debug(f"Stored ROI metadata for {cropped_filename}")

            except Exception as e:
                logging.error(f"Failed to save cropped image {full_output_path} or store its metadata: {e}")
        else:
            logging.warning(f"Highest confidence detection in {current_image_identifier} (conf: {max_confidence:.2f}): Invalid crop dimensions ({crop_x1, crop_y1, crop_x2, crop_y2}). Skipping crop.")
    else:
        logging.info(f"No valid highest confidence detection found in {current_image_identifier}, though detections were present.")


    if detections_saved_count > 0:
        logging.info(f"Saved 1 highest confidence cropped image from {current_image_identifier}.")
    elif hailo_detections: 
        logging.warning(f"Found {len(hailo_detections)} detections in {current_image_identifier}, but the highest confidence one was not saved/processed.")
    return Gst.PadProbeReturn.OK


# -----------------------------------------------------------------------------------------------
# GStreamerApp class
# -----------------------------------------------------------------------------------------------
class App:
    def __init__(self):
        setproctitle.setproctitle("Hailo Python App")
        signal.signal(signal.SIGINT, self.shutdown_signal_handler) # Changed to new handler name

        tappas_post_process_dir = os.environ.get('TAPPAS_POST_PROC_DIR', '')
        if tappas_post_process_dir == '':
            print("TAPPAS_POST_PROC_DIR environment variable is not set. Please set it to by sourcing setup_env.sh")
            exit(1)
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.postprocess_dir = tappas_post_process_dir
        self.pipeline = None
        self.loop = None
        self.image_size = (2304, 1296)

        logging.basicConfig(
            filename='app.log',
            level=logging.DEBUG,
            format='%(asctime)s %(levelname)s: %(message)s', # Added timestamp
            filemode='w'
        )

        self.pipeline_latency = 0
        self.batch_size = 2
        self.video_width = 2304
        self.video_height = 1296
        self.video_format = "RGB"
        self.threads = []

        os.makedirs(os.path.join(OUTPUT_CROPPED_BASE_DIR, "left"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_CROPPED_BASE_DIR, "right"), exist_ok=True)
        
        self.create_pipeline()
        self.error_occurred = False

    def on_fps_measurement(self, sink, fps, droprate, avgfps):
        return True

    def create_pipeline(self):
        Gst.init(None)
        pipeline_string = self.get_pipeline_string()
        try:
            self.pipeline = Gst.parse_launch(pipeline_string)
            print("Pipeline parsed and launched")
        except Exception as e:
            print(f"Error parsing pipeline: {e}")
            print(f"Pipeline string: {pipeline_string}")
            sys.exit(1)
        self.loop = GLib.MainLoop()

    def bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End-of-stream received on bus.")
            self.on_eos()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error on bus: {err}, {debug}")
            logging.error(f"GStreamer bus error: {err} - {debug}")
            self.error_occurred = True # Mark error
            self.perform_shutdown() # Initiate shutdown on error
        elif t == Gst.MessageType.QOS:
            qos_element = "UnknownElement"
            if message.src:
                 qos_element = message.src.get_name()
            # logging.debug(f"QoS message received from {qos_element}") # Too verbose for info
        return True

    def on_eos(self):
        logging.info("EOS signaled. Initiating shutdown sequence.")
        self.perform_shutdown()
    
    def shutdown_signal_handler(self, signum, frame):
        print("Shutdown signal (Ctrl-C) received. Initiating graceful shutdown...")
        logging.info("Shutdown signal (Ctrl-C) received. Initiating graceful shutdown...")
        # Ensure perform_shutdown is called from GLib idle_add if in a different thread context
        # or if it manipulates GStreamer objects that should be handled by the main loop thread.
        # For direct signal handling that calls GStreamer state changes, it might be okay,
        # but GLib.idle_add is safer for complex GStreamer interactions from signals.
        GLib.idle_add(self.perform_shutdown)
        # Replace the handler with default to allow force quit on second Ctrl-C
        signal.signal(signal.SIGINT, signal.SIG_DFL) 

    def perform_shutdown(self):
        print("Performing shutdown operations...")
        if self.pipeline:
            print("Setting pipeline to PAUSED...")
            self.pipeline.set_state(Gst.State.PAUSED)
            # GLib.usleep(100000) # 0.1s, might not be strictly necessary if EOS is handled
            
            print("Setting pipeline to READY...")
            self.pipeline.set_state(Gst.State.READY)
            # GLib.usleep(100000)

            print("Setting pipeline to NULL...")
            self.pipeline.set_state(Gst.State.NULL)
        
        # Save ROI metadata
        if global_roi_metadata_dict:
            metadata_output_path = os.path.join(OUTPUT_CROPPED_BASE_DIR, ROI_METADATA_FILENAME)
            try:
                with open(metadata_output_path, 'w') as f:
                    json.dump(global_roi_metadata_dict, f, indent=4)
                logging.info(f"ROI metadata saved to {metadata_output_path}")
                print(f"ROI metadata saved to {metadata_output_path}")
            except Exception as e:
                logging.error(f"Failed to save ROI metadata: {e}")
                print(f"Error: Failed to save ROI metadata: {e}")
        else:
            logging.info("No ROI metadata to save.")
            print("No ROI metadata to save.")

        if self.loop and self.loop.is_running():
            print("Quitting GLib MainLoop...")
            self.loop.quit()
        else:
            print("GLib MainLoop not running or already quit.")
        
        # Threads should ideally join after pipeline is NULL and loop is quit
        # This part is moved to the end of run() method for cleaner shutdown sequence.
        # print("Joining threads...")
        # for t in self.threads:
        #     if t.is_alive():
        #         t.join(timeout=1.0) # Add timeout
        # print("Threads joined.")

        print("Shutdown operations complete.")


    def get_pipeline_string(self):
        return PIPELINE_STRING(display=False)

    def run(self):
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)

        identity_1 = self.pipeline.get_by_name("identity_callback")
        if identity_1 is None:
            print("Warning: identity_callback element not found.")
            logging.warning("identity_callback element not found in the pipeline.")
        else:
            identity_1_pad = identity_1.get_static_pad("src")
            identity_1_pad.add_probe(Gst.PadProbeType.BUFFER, camera_callback)
            print("Pad probe added to identity_callback.")

        disable_qos(self.pipeline)

        picam_thread = threading.Thread(target=picamera_thread, args=[self.pipeline])
        self.threads.append(picam_thread)
        picam_thread.start()

        self.pipeline.set_state(Gst.State.PAUSED)
        # new_latency = self.pipeline_latency * Gst.MSECOND # Gst.MSECOND is 1_000_000 ns
        # self.pipeline.set_latency(new_latency)
        # Setting latency to 0 for non-live sources or if precise sync isn't critical
        self.pipeline.set_latency(0) 

        print("Setting pipeline to PLAYING...")
        self.pipeline.set_state(Gst.State.PLAYING)
        
        try:
            print("Running GLib MainLoop...")
            self.loop.run()
        except KeyboardInterrupt: # This might not be caught if SIGINT is handled by shutdown_signal_handler
            print("KeyboardInterrupt caught in loop.run(). Initiating shutdown.")
            logging.warning("KeyboardInterrupt caught in loop.run(). Initiating shutdown.")
            self.perform_shutdown() # Ensure shutdown is called
        finally:
            print("GLib MainLoop finished.")
            # Ensure all threads are joined after the loop has finished and pipeline is down
            print("Final cleanup: Joining threads...")
            for t in self.threads:
                if t.is_alive():
                    print(f"Joining thread {t.name}...")
                    t.join(timeout=2.0) # Increased timeout for safety
                    if t.is_alive():
                        print(f"Warning: Thread {t.name} did not join after timeout.")
                        logging.warning(f"Thread {t.name} did not join after timeout.")
            print("All threads joined.")

            if self.error_occurred:
                print("Exiting with error code 1.", file=sys.stderr)
                sys.exit(1)
            else:
                print("Exiting cleanly with code 0.")
                sys.exit(0)

def picamera_thread(pipeline):
    appsrc = pipeline.get_by_name("app_source")
    appsrc.set_property("is-live", True)
    appsrc.set_property("format", Gst.Format.TIME)

    # Load calibration data
    calib_data = None
    try:
        if os.path.exists(CALIBRATION_FILE_PATH):
            calib_data = CalibrationData(CALIBRATION_FILE_PATH)
            logging.info(f"Successfully loaded calibration data from {CALIBRATION_FILE_PATH}")
        else:
            logging.error(f"Calibration file not found at {CALIBRATION_FILE_PATH}. Proceeding without rectification.")
    except Exception as e:
        logging.error(f"Failed to load calibration data from {CALIBRATION_FILE_PATH}: {e}. Proceeding without rectification.")
        calib_data = None


    image_width = 2304 # Assuming this from your class App init, adjust if needed
    image_height = 1296 # Assuming this from your class App init, adjust if needed
    image_format = "RGB" # Assuming this from your class App init, adjust if needed

    # Path to images - adjust base path as necessary
    base_image_path = "data/basketball_images"
    left_image_files = sorted(glob.glob(os.path.join(base_image_path, "left", "*.jpg"))) # Assuming jpg, adjust if other format
    right_image_files = sorted(glob.glob(os.path.join(base_image_path, "right", "*.jpg")))

    if not left_image_files or not right_image_files:
        logging.error(f"No image files found in {os.path.join(base_image_path, 'left')} or {os.path.join(base_image_path, 'right')}. Check paths and file types.")
        appsrc.end_of_stream() # Signal EOS if no files
        return
    
    if len(left_image_files) != len(right_image_files):
        logging.warning("Warning: Mismatch in number of left and right images. Processing shorter list.")
        num_pairs = min(len(left_image_files), len(right_image_files))
        logging.info(f"Number of left images: {len(left_image_files)}, number of right images: {len(right_image_files)}. Processing {num_pairs} pairs.")
        left_image_files = left_image_files[:num_pairs]
        right_image_files = right_image_files[:num_pairs]

    # The GStreamer pipeline caps should ideally match the rectified image dimensions.
    # If rectification changes dimensions, image_width and image_height might need adjustment.
    # For now, assuming rectification preserves original dimensions.
    appsrc.set_property(
        "caps",
        Gst.Caps.from_string(
            f"video/x-raw, format={image_format}, width={image_width}, height={image_height}, "
            f"framerate=10/1, pixel-aspect-ratio=1/1" 
        )
    )
    buffer_idx = 0 
    fps = 10 

    global pts_to_filepath_map 
    pts_to_filepath_map.clear() # Clear any previous mappings if thread restarts

    for left_img_path, right_img_path in zip(left_image_files, right_image_files):
        left_image_orig = cv2.imread(left_img_path)
        if left_image_orig is None:
            logging.error(f"Failed to read left image: {left_img_path}")
            continue
        
        right_image_orig = cv2.imread(right_img_path)
        if right_image_orig is None:
            logging.error(f"Failed to read right image: {right_img_path}")
            continue

        # Rectify images if calibration data is available
        rect_left_img, rect_right_img = left_image_orig, right_image_orig # Default to original if no calibration
        if calib_data:
            try:
                # Ensure maps are initialized for current image size
                if calib_data.image_size != left_image_orig.shape[1::-1]: # (width, height)
                    calib_data.init_rectification_maps(left_image_orig.shape[1::-1])
                
                rect_left_img, rect_right_img = calib_data.rectify_image_pair(left_image_orig, right_image_orig)
                
                # Update image_width and image_height if rectification changes them and it's the first pair
                # This is important if the appsrc caps were set based on an assumption.
                # However, initUndistortRectifyMap with P matrices usually maps to the same image_size passed.
                # So, this explicit update might not be needed if image_width/height already match raw.
                # For safety, let's assume rectified images dimensions match image_width, image_height.
                # If not, a resize is needed or caps must be set dynamically.

            except Exception as e:
                logging.error(f"Error during rectification for {left_img_path}/{right_img_path}: {e}. Using original images.")
                rect_left_img, rect_right_img = left_image_orig, right_image_orig


        # Ensure image is in the correct format (e.g., RGB) and dimensions
        # OpenCV loads as BGR. If pipeline expects RGB, convert.
        # The pipeline string specifies format=RGB for appsrc, so conversion is needed.
        
        # Process left image
        left_image_processed = cv2.cvtColor(rect_left_img, cv2.COLOR_BGR2RGB)
        if left_image_processed.shape[1] != image_width or left_image_processed.shape[0] != image_height:
            left_image_processed = cv2.resize(left_image_processed, (image_width, image_height))
        
        left_data = left_image_processed.tobytes()
        buffer_left = Gst.Buffer.new_wrapped(left_data)
        buffer_left.pts = Gst.util_uint64_scale_int(buffer_idx, Gst.SECOND, fps)
        buffer_left.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, fps)
        buffer_left.set_flags(Gst.BufferFlags(1))  # Flag as left camera
        pts_to_filepath_map[buffer_left.pts] = left_img_path # Store original path
        ret = appsrc.emit('push-buffer', buffer_left)
        if ret != Gst.FlowReturn.OK:
            logging.error(f"Failed to push left buffer for {left_img_path}: {ret}")
            break
        logging.debug(f"Pushed left frame {os.path.basename(left_img_path)} (rectified) with PTS {buffer_left.pts}")
        buffer_idx += 1

        # Process right image
        right_image_processed = cv2.cvtColor(rect_right_img, cv2.COLOR_BGR2RGB)
        if right_image_processed.shape[1] != image_width or right_image_processed.shape[0] != image_height:
            right_image_processed = cv2.resize(right_image_processed, (image_width, image_height))
        
        right_data = right_image_processed.tobytes()
        buffer_right = Gst.Buffer.new_wrapped(right_data)
        buffer_right.pts = Gst.util_uint64_scale_int(buffer_idx, Gst.SECOND, fps) 
        buffer_right.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, fps)
        buffer_right.set_flags(Gst.BufferFlags(2))  # Flag as right camera
        pts_to_filepath_map[buffer_right.pts] = right_img_path # Store original path
        ret = appsrc.emit('push-buffer', buffer_right)
        if ret != Gst.FlowReturn.OK:
            logging.error(f"Failed to push right buffer for {right_img_path}: {ret}")
            break
        logging.debug(f"Pushed right frame {os.path.basename(right_img_path)} (rectified) with PTS {buffer_right.pts}")
        buffer_idx += 1
        
        time.sleep(1/fps) 

    logging.info("All images processed by picamera_thread, sending EOS to appsrc.")
    appsrc.end_of_stream()


if __name__ == "__main__":
    app = App()
    app.run()