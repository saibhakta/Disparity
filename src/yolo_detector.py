import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import threading
import time
import sys
import os # For TAPPAS_POST_PROC_DIR
import cv2 # Added cv2 import
import hailo
import argparse # Added for command-line arguments


DEFAULT_INPUT_WIDTH = 2304
DEFAULT_INPUT_HEIGHT = 1296

def PIPELINE_STRING(display=True):
    if display:
        return "appsrc name=app_source is-live=true leaky-type=downstream max-buffers=1 ! video/x-raw, format=RGB, width=2304, height=1296 !  queue name=source_scale_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoscale name=source_videoscale n-threads=2 ! queue name=source_convert_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoconvert n-threads=3 name=source_convert qos=false ! video/x-raw, pixel-aspect-ratio=1/1, format=RGB, width=2304, height=1296  ! queue name=inference_wrapper_input_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailocropper name=inference_wrapper_crop so-path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true hailoaggregator name=inference_wrapper_agg inference_wrapper_crop. ! queue name=inference_wrapper_bypass_q leaky=downstream max-size-buffers=10 max-size-bytes=0 max-size-time=0  ! inference_wrapper_agg.sink_0 inference_wrapper_crop. ! queue name=inference_scale_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoscale name=inference_videoscale n-threads=2 qos=false ! queue name=inference_convert_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! video/x-raw, pixel-aspect-ratio=1/1 ! videoconvert name=inference_videoconvert n-threads=2 ! queue name=inference_hailonet_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailonet name=inference_hailonet hef-path=/home/sai/Disparity/resources/yolov11n.hef batch-size=2  vdevice-group-id=1 nms-score-threshold=0.3 nms-iou-threshold=0.45 output-format-type=HAILO_FORMAT_TYPE_FLOAT32 force-writable=true  ! queue name=inference_hailofilter_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailofilter name=inference_hailofilter so-path=/home/sai/Disparity/resources/libyolo_hailortpp_postprocess.so   function-name=filter_letterbox  qos=false ! queue name=inference_output_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0   ! inference_wrapper_agg.sink_1 inference_wrapper_agg. ! queue name=inference_wrapper_output_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0   ! queue name=identity_callback_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! identity name=identity_callback ! queue name=hailo_display_overlay_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailooverlay name=hailo_display_overlay  ! queue name=hailo_display_videoconvert_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoconvert name=hailo_display_videoconvert n-threads=2 qos=false ! queue name=hailo_display_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! fpsdisplaysink name=hailo_display video-sink=autovideosink sync=false text-overlay=False signal-fps-measurements=true"
    
    # No display
    return "appsrc name=app_source is-live=true leaky-type=upstream max-buffers=1 ! video/x-raw, format=RGB, width=2304, height=1296 !  queue name=source_scale_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoscale name=source_videoscale n-threads=2 ! queue name=source_convert_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoconvert n-threads=3 name=source_convert qos=false ! video/x-raw, pixel-aspect-ratio=1/1, format=RGB, width=2304, height=1296  ! queue name=inference_wrapper_input_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailocropper name=inference_wrapper_crop so-path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true hailoaggregator name=inference_wrapper_agg inference_wrapper_crop. ! queue name=inference_wrapper_bypass_q leaky=downstream max-size-buffers=10 max-size-bytes=0 max-size-time=0  ! inference_wrapper_agg.sink_0 inference_wrapper_crop. ! queue name=inference_scale_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoscale name=inference_videoscale n-threads=2 qos=false ! queue name=inference_convert_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! video/x-raw, pixel-aspect-ratio=1/1 ! videoconvert name=inference_videoconvert n-threads=2 ! queue name=inference_hailonet_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailonet name=inference_hailonet hef-path=/home/sai/Disparity/resources/yolov11n.hef batch-size=2  vdevice-group-id=1 nms-score-threshold=0.3 nms-iou-threshold=0.45 output-format-type=HAILO_FORMAT_TYPE_FLOAT32 force-writable=true  ! queue name=inference_hailofilter_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailofilter name=inference_hailofilter so-path=/home/sai/Disparity/resources/libyolo_hailortpp_postprocess.so   function-name=filter_letterbox  qos=false ! queue name=inference_output_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0   ! inference_wrapper_agg.sink_1 inference_wrapper_agg. ! queue name=inference_wrapper_output_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0   ! queue name=identity_callback_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! identity name=identity_callback ! fakesink"



class YoloDetector:
    def __init__(self, 
                 input_width=DEFAULT_INPUT_WIDTH,
                 input_height=DEFAULT_INPUT_HEIGHT,
                 target_class_id=0, min_confidence=0.3):
        
        Gst.init(None)
    
        self.input_width = input_width
        self.input_height = input_height
        self.target_class_id = target_class_id # Assuming basketball is class 0 in your YOLO model
        self.min_confidence = min_confidence

        self.pipeline = None
        self.appsrc = None
        self.detections_lock = threading.Lock()
        self.last_detections = [] # Stores (bbox_xywh, confidence)
        self.gst_thread = None
        self.loop = None
        self._running = False
        self._gst_ready_event = threading.Event()


    def _gst_loop(self):
        self.loop = GLib.MainLoop()
        self.loop.run() # This blocks until self.loop.quit() is called
        print("GStreamer loop exited.")
        self._running = False
        
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
            self.appsrc = None
        print("YOLO Detector GStreamer pipeline cleaned up.")

    def start(self):
        if self._running:
            print("YOLO Detector already running.")
            return

        pipeline_str = PIPELINE_STRING(display=True)
        print("Initializing GStreamer pipeline for YOLO detection...")
        
        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
        except GLib.Error as e:
            print(f"Failed to parse GStreamer pipeline: {e}")
            print("Ensure GStreamer, Hailo plugins, and specified .hef/.so files are correctly installed and accessible.")
            print("Pipeline string was:")
            print(pipeline_str)
            return

        self.appsrc = self.pipeline.get_by_name("app_source")
        if not self.appsrc:
            print("Error: appsrc element not found in pipeline.")
            self.pipeline.set_state(Gst.State.NULL) # Clean up
            self.pipeline = None
            return

        # Set up appsrc properties
        # Caps should match what you're pushing
        # The pipeline string already defines caps for appsrc, so explicit setting might not be needed
        # unless pushing different format/size initially.
        # For this project, input is assumed to be rectified image from file.
        
        # Setup bus message handling
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        # Setup callback for detections
        identity_callback = self.pipeline.get_by_name("identity_callback")
        if identity_callback:
            id_pad = identity_callback.get_static_pad("src")
            id_pad.add_probe(Gst.PadProbeType.BUFFER, self._on_detections_probe)
        else:
            print("Warning: identity_callback element not found in pipeline.")

        # Start the GStreamer pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
        self._running = True
        self._gst_ready_event.clear()

        # Run GLib main loop in a separate thread
        self.gst_thread = threading.Thread(target=self._gst_loop, daemon=True)
        self.gst_thread.start()
        
        print("Waiting for GStreamer pipeline to be ready...")
        # Wait for a short period or use an event signaled by the PLAYING state
        ready = self._gst_ready_event.wait(timeout=5.0) # Wait up to 5 seconds
        if not ready:
            print("Warning: GStreamer pipeline might not have reached PLAYING state in time.")
        else:
            print("GStreamer pipeline is PLAYING.")


    def stop(self):
        if not self._running:
            print("YOLO Detector not running.")
            return
        print("Stopping YOLO Detector...")
        self._running = False # Signal threads to stop pushing data

        if self.loop and self.loop.is_running():
            self.loop.quit()
        
        if self.gst_thread and self.gst_thread.is_alive():
            self.gst_thread.join(timeout=5.0) # Wait for GStreamer thread to finish
            if self.gst_thread.is_alive():
                print("Warning: GStreamer thread did not terminate cleanly.")
        
        # Pipeline state set to NULL is handled in _gst_loop on exit or here if loop didn't run
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
            self.appsrc = None
        print("YOLO Detector stopped.")


    def _on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("YOLO Detector: End-of-stream")
            if self.loop: self.loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"YOLO Detector Error: {err}, {debug}")
            if self.loop: self.loop.quit()
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending_state = message.parse_state_changed()
                print(f"YOLO Pipeline state changed from {old_state.value_nick} to {new_state.value_nick}")
                if new_state == Gst.State.PLAYING:
                    self._gst_ready_event.set() # Signal that pipeline is ready
        return True


    def _on_detections_probe(self, pad, info):
        buffer = info.get_buffer()
        if buffer:
            detections = []
            try:
                # Get HailoROI from buffer (assuming Hailo GStreamer plugins are used)
                roi = hailo.get_roi_from_buffer(buffer)
                hailo_dets = roi.get_objects_typed(hailo.HAILO_DETECTION)

                for det in hailo_dets:
                    if det.get_label() == "basketball" or det.get_class_id() == self.target_class_id: # Check label or class ID
                        confidence = det.get_confidence()
                        if confidence >= self.min_confidence:
                            bbox = det.get_bbox() # This is HailoBBox
                            # Convert normalized bbox (0.0-1.0) to absolute pixel values (x,y,w,h)
                            # Need image dimensions for this. Assuming input is 2304x1296 as per pipeline.
                            # The hailocropper might resize *before* hailonet.
                            # The bbox from postprocess is usually relative to the network input size (e.g. 640x640).
                            # The `filter_letterbox` function in postprocess typically adjusts coords
                            # back to the original image size *before letterboxing/cropping*.
                            # So, bbox.xmin() etc. should be relative to the 2304x1296 frame pushed to appsrc.
                            img_w, img_h = self.input_width, self.input_height # Use stored params
                            
                            x = bbox.xmin() * img_w
                            y = bbox.ymin() * img_h
                            w = bbox.width() * img_w
                            h = bbox.height() * img_h
                            detections.append(((int(x), int(y), int(w), int(h)), confidence))
            except Exception as e:
                print(f"Error processing detections in probe: {e}")
                # This might happen if buffer doesn't contain Hailo metadata as expected

            with self.detections_lock:
                self.last_detections = detections
        return Gst.PadProbeReturn.OK


    def detect_basketball_roi(self, image_np: np.ndarray):
        """
        Detects basketball in the given image and returns the ROI.
        Args:
            image_np: NumPy array of the image (RGB format, HxWxC).
        Returns:
            A tuple (x, y, width, height) for the ROI, or None if not found.
        """
        if not self._running or not self.appsrc or not self.pipeline or self.pipeline.get_state(0)[1] != Gst.State.PLAYING:
            print("YOLO Detector is not running or pipeline not in PLAYING state.")
            # Try to start it if it's not running at all
            if not self._running:
                print("Attempting to start YOLO detector...")
                self.start()
                if not self._running:
                    return None # Failed to start
                # Wait a bit for pipeline to be ready after manual start
                time.sleep(1.0) if not self._gst_ready_event.is_set() else None


        h, w, c = image_np.shape
        if not (h == self.input_height and w == self.input_width and c == 3):
             print(f"Warning: Input image shape {image_np.shape} does not match pipeline appsrc caps ({self.input_width}x{self.input_height} RGB). Resizing.")
             # This resize might affect detection quality if aspect ratio changes significantly without letterboxing
             # For evaluation, it's best if input images are already at the expected GStreamer pipeline input size.
             image_np = cv2.resize(image_np, (self.input_width, self.input_height))


        # Convert to Gst.Buffer
        gst_buffer = Gst.Buffer.new_wrapped(image_np.tobytes())
        
        # Set timestamp for the buffer if appsrc `do-timestamp=true`
        pts = Gst.util_uint64_scale_int(int(time.monotonic() * 10**9), Gst.SECOND, 1) # Nanoseconds
        gst_buffer.pts = pts
        gst_buffer.duration = Gst.CLOCK_TIME_NONE


        # Push buffer to appsrc
        with self.detections_lock:
            self.last_detections = [] # Clear previous detections for this frame

        ret = self.appsrc.emit("push-buffer", gst_buffer)
        if ret != Gst.FlowReturn.OK:
            print(f"Error pushing buffer to appsrc: {ret}")
            return None

        # Wait a short time for processing and callback to update last_detections
        # This is a polling approach; a more robust way might involve condition variables.
        # Or, if this function is called infrequently, a slightly longer fixed wait is okay.
        time.sleep(0.05) # Adjust as needed, depends on pipeline latency

        with self.detections_lock:
            if self.last_detections:
                # Find the detection with the highest confidence
                best_detection = max(self.last_detections, key=lambda item: item[1])
                return best_detection[0] # Return (x,y,w,h)
        return None

    def __del__(self):
        self.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect basketballs in images and save cropped versions.")
    parser.add_argument("--input_folder", type=str, default="data/basketball_images",
                        help="Folder containing images to process (e.g., data/basketball_images). Expects 'left' and 'right' subfolders.")
    parser.add_argument("--output_folder", type=str, default="data/cropped_images",
                        help="Folder to save cropped images (e.g., data/cropped_images). Will replicate 'left'/'right' subfolder structure.")
    args = parser.parse_args()

    input_base_folder = args.input_folder
    output_base_folder = args.output_folder

    print(f"Starting YOLO Detector image processing...")
    print(f"Input folder: {input_base_folder}")
    print(f"Output folder: {output_base_folder}")

    detector = YoloDetector()
    detector.start()

    if not detector._running:
        print("Failed to start detector. Exiting.")
        sys.exit(1)

    processed_files = 0
    detected_and_saved_files = 0
    skipped_files = 0

    # Process images in 'left' and 'right' subdirectories
    for subfolder_name in ["left", "right"]:
        input_subfolder = os.path.join(input_base_folder, subfolder_name)
        output_subfolder = os.path.join(output_base_folder, subfolder_name)

        if not os.path.isdir(input_subfolder):
            print(f"Warning: Input subfolder {input_subfolder} not found. Skipping.")
            continue

        os.makedirs(output_subfolder, exist_ok=True)
        
        print(f"Processing images in {input_subfolder}...")

        for filename in os.listdir(input_subfolder):
            if not (filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))):
                if filename != ".DS_Store": # Common macOS file, safe to ignore
                     print(f"Skipping non-image file: {filename} in {input_subfolder}")
                continue

            input_image_path = os.path.join(input_subfolder, filename)
            output_image_path = os.path.join(output_subfolder, filename) # Save with the same name

            processed_files += 1
            print(f"Processing {input_image_path}... ", end="")

            if os.path.exists(output_image_path):
                print(f"Skipped (already exists: {output_image_path})")
                skipped_files += 1
                continue

            try:
                # Read the image using OpenCV
                image_np = cv2.imread(input_image_path)
                if image_np is None:
                    print(f"Failed to read image {input_image_path}. Skipping.")
                    continue
                
                # Image needs to be in RGB for the detector if cv2 reads in BGR by default
                image_np_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

                roi = detector.detect_basketball_roi(image_np_rgb)

                if roi:
                    x, y, w, h = roi
                    # Ensure ROI coordinates are valid
                    if w > 0 and h > 0 and x >= 0 and y >= 0 and (x + w) <= image_np.shape[1] and (y + h) <= image_np.shape[0]:
                        cropped_image = image_np[y:y+h, x:x+w]
                        cv2.imwrite(output_image_path, cropped_image)
                        print(f"Detected. Saved cropped image to {output_image_path}")
                        detected_and_saved_files +=1
                    else:
                        print(f"Detected, but ROI {roi} is invalid for image size {image_np.shape[:2]}. Not saving.")
                else:
                    print("No basketball detected.")
            except Exception as e:
                print(f"Error processing file {input_image_path}: {e}")

    print("\nProcessing summary:")
    print(f"Total files considered for processing: {processed_files}")
    print(f"Files skipped (already existed in output): {skipped_files}")
    print(f"Files where basketball was detected and cropped image saved: {detected_and_saved_files}")
    
    print("Stopping YOLO detector...")
    detector.stop()
    print("YOLO Detector image processing finished.")
