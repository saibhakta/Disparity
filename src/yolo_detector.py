import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import threading
import time
import sys
import os # For TAPPAS_POST_PROC_DIR
import cv2 # Added cv2 import

# Attempt to import Hailo specific modules
try:
    import hailo
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    # print("Warning: Hailo Python module not found. YOLO detector will not function.")

# Import GStreamer helpers from your original codebase if they are in PYTHONPATH
# Or, include them directly/modify paths. For this standalone project, let's assume
# they might be needed if we were to parse raw buffers for debugging.
# from gst_pipeline.gstreamer_helpers import get_caps_from_pad, get_numpy_from_buffer

# This is the GStreamer pipeline string for fast detection from your original codebase.
# IMPORTANT: Ensure the .hef path and .so path are correct for this project's environment.
# The original paths are /home/sai/Robot/resources/yolov11n.hef and
# /path/to/libyolo_hailortpp_postprocess.so (which was a placeholder).
# You need to replace "/path/to/libyolo_hailortpp_postprocess.so" with the actual path
# to your Hailo postprocess .so file for YOLOv11.
# e.g., /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstha Mypostprocess.so or similar.
# A common path for TAPPAS postprocess .so is often found via TAPPAS_POST_PROC_DIR
# or within the Hailo SDK installation.

# For this project, let's make the .hef path configurable.
DEFAULT_HEF_PATH = "resources/yolov11n.hef" # Relative to project root
# The postprocess .so path is more tricky. It's often system-installed.
# Let's try to use TAPPAS_POST_PROC_DIR if set, otherwise provide a placeholder.
TAPPAS_POST_PROC_DIR = os.environ.get('TAPPAS_POST_PROC_DIR', '/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/')
DEFAULT_POSTPROCESS_SO_PATH = os.path.join(TAPPAS_POST_PROC_DIR, "libyolo_hailortpp_postprocess.so") # Check actual name
DEFAULT_CROPPER_SO_PATH = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so" # Default cropper path
DEFAULT_INPUT_WIDTH = 2304
DEFAULT_INPUT_HEIGHT = 1296


def get_pipeline_string_fast_detection(hef_path, postprocess_so_path, cropper_so_path, input_width, input_height):
    # Check if the postprocess .so file exists, otherwise GStreamer will fail to parse.
    if not os.path.exists(postprocess_so_path):
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"WARNING: Postprocess SO file NOT FOUND: {postprocess_so_path}")
        print(f"The YOLO GStreamer pipeline will likely FAIL.")
        print(f"Please ensure the path is correct or the Hailo environment is set up.")
        print(f"Searched based on TAPPAS_POST_PROC_DIR: {TAPPAS_POST_PROC_DIR}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Fallback to a generic name if a common one exists or provide an obvious placeholder.
        # For now, we'll use the provided path and let GStreamer error out if it's wrong.

    return (
        f"appsrc name=app_source is-live=true do-timestamp=true format=time leaky-type=upstream max-buffers=2 ! " 
        f"video/x-raw, format=RGB, width={input_width}, height={input_height}, framerate=10/1 ! " 
        "queue name=pre_crop_q leaky=downstream max-size-buffers=2 ! "
        # Hailocropper might resize. YOLOv11n often expects 640x640.
        # If input to hailocropper is already desired size, `use-letterbox=false` might be better.
        # If hailocropper is resizing to model input, ensure `internal-offset=true` is what you want.
        f"hailocropper name=inference_wrapper_crop so-path={cropper_so_path} function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true ! "
        "queue name=pre_infer_q leaky=downstream max-size-buffers=2 ! "
        f"hailonet name=inference_hailonet hef-path={hef_path} batch-size=1 nms-score-threshold=0.3 nms-iou-threshold=0.45 output-format-type=HAILO_FORMAT_TYPE_FLOAT32 force-writable=true ! " # batch-size=1 for single image processing
        "queue name=pre_postproc_q leaky=downstream max-size-buffers=2 ! "
        f"hailofilter name=inference_hailofilter so-path={postprocess_so_path} function-name=filter_letterbox qos=false ! "
        "identity name=identity_callback ! " # Callback to get detections
        "fakesink name=fakesink sync=false async=false" # Ensure fakesink doesn't block
    )


class YoloDetector:
    def __init__(self, hef_path=DEFAULT_HEF_PATH, 
                 postprocess_so_path=DEFAULT_POSTPROCESS_SO_PATH, 
                 cropper_so_path=DEFAULT_CROPPER_SO_PATH,
                 input_width=DEFAULT_INPUT_WIDTH,
                 input_height=DEFAULT_INPUT_HEIGHT,
                 target_class_id=0, min_confidence=0.3):
        if not HAILO_AVAILABLE:
            raise ImportError("Hailo Python module is not available. YoloDetector cannot be initialized.")
        
        Gst.init(None)
        self.hef_path = hef_path
        self.postprocess_so_path = postprocess_so_path
        self.cropper_so_path = cropper_so_path
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

        pipeline_str = get_pipeline_string_fast_detection(
            self.hef_path, 
            self.postprocess_so_path,
            self.cropper_so_path,
            self.input_width,
            self.input_height
        )
        print("Initializing GStreamer pipeline for YOLO detection...")
        print(f"Using HEF: {self.hef_path}")
        print(f"Using Postprocess SO: {self.postprocess_so_path}")
        
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
    # Example Usage (requires a running X server for GStreamer video sinks if pipeline had one)
    # This example will use fakesink, so no visual output from GStreamer itself.
    
    if not HAILO_AVAILABLE:
        print("Hailo SDK not available, cannot run YoloDetector example.")
        sys.exit(1)

    print("Starting YOLO Detector example...")
    # You might need to adjust these paths based on your setup for the example
    # Ensure 'resources/yolov11n.hef' exists or provide the correct path.
    custom_hef_path = "resources/yolov11n.hef" # Adjust if your .hef is elsewhere
    if not os.path.exists(custom_hef_path):
        print(f"Error: HEF file not found at {custom_hef_path}. Please place it there or update path.")
        sys.exit(1)
    
    # The postprocess_so_path is critical. Check Hailo examples or SDK for the correct one.
    # This is just a guess based on common TAPPAS structure.
    custom_so_path = os.path.join(os.environ.get('TAPPAS_POST_PROC_DIR', '/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/'), 'libyolo_hailortpp_postprocess.so')


    detector = YoloDetector(hef_path=custom_hef_path, postprocess_so_path=custom_so_path)
    detector.start()

    if not detector._running:
        print("Failed to start detector. Exiting example.")
        sys.exit(1)

    # Create a dummy image (2304x1296 RGB)
    dummy_image = np.random.randint(0, 255, size=(1296, 2304, 3), dtype=np.uint8)
    # Put a colored square to simulate a basketball for visual debugging if you were to save/show the image
    cv2.rectangle(dummy_image, (500, 500), (700, 700), (0, 165, 255), -1) # Orange-ish

    print("Detecting ROI in dummy image...")
    roi = detector.detect_basketball_roi(dummy_image)

    if roi:
        print(f"Detected basketball ROI: {roi}")
        # For visualization:
        # import cv2
        # from utils import draw_roi, display_image
        # img_with_roi = draw_roi(cv2.cvtColor(dummy_image, cv2.COLOR_RGB2BGR), roi) # Convert to BGR for OpenCV display
        # display_image("Dummy Image with ROI", img_with_roi)
        # cv2.destroyAllWindows()
    else:
        print("No basketball detected in the dummy image.")

    print("Stopping YOLO detector...")
    detector.stop()
    print("YOLO Detector example finished.")

    # For visualization (needs cv2 and utils):
    # from .utils import draw_roi, display_image 
    # import cv2

# Ensure imports for main example are also present if uncommented
# import cv2
# import sys
# from .utils import draw_roi, display_image # If example uses these