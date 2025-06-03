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

def PIPELINE_STRING(display=True):
    if display:
        return "appsrc name=app_source is-live=true leaky-type=downstream max-buffers=1 ! video/x-raw, format=RGB, width=2304, height=1296 !  queue name=source_scale_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoscale name=source_videoscale n-threads=2 ! queue name=source_convert_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoconvert n-threads=3 name=source_convert qos=false ! video/x-raw, pixel-aspect-ratio=1/1, format=RGB, width=2304, height=1296  ! queue name=inference_wrapper_input_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailocropper name=inference_wrapper_crop so-path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true hailoaggregator name=inference_wrapper_agg inference_wrapper_crop. ! queue name=inference_wrapper_bypass_q leaky=downstream max-size-buffers=10 max-size-bytes=0 max-size-time=0  ! inference_wrapper_agg.sink_0 inference_wrapper_crop. ! queue name=inference_scale_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoscale name=inference_videoscale n-threads=2 qos=false ! queue name=inference_convert_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! video/x-raw, pixel-aspect-ratio=1/1 ! videoconvert name=inference_videoconvert n-threads=2 ! queue name=inference_hailonet_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailonet name=inference_hailonet hef-path=/home/sai/Robot/resources/yolov11n.hef batch-size=2  vdevice-group-id=1 nms-score-threshold=0.3 nms-iou-threshold=0.45 output-format-type=HAILO_FORMAT_TYPE_FLOAT32 force-writable=true  ! queue name=inference_hailofilter_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailofilter name=inference_hailofilter so-path=/home/sai/Robot/venv_hailo_rpi5_examples/lib/python3.11/site-packages/hailo_apps_infra/../resources/libyolo_hailortpp_postprocess.so   function-name=filter_letterbox  qos=false ! queue name=inference_output_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0   ! inference_wrapper_agg.sink_1 inference_wrapper_agg. ! queue name=inference_wrapper_output_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0   ! queue name=identity_callback_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! identity name=identity_callback ! queue name=hailo_display_overlay_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailooverlay name=hailo_display_overlay  ! queue name=hailo_display_videoconvert_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoconvert name=hailo_display_videoconvert n-threads=2 qos=false ! queue name=hailo_display_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! fpsdisplaysink name=hailo_display video-sink=autovideosink sync=false text-overlay=False signal-fps-measurements=true"
    
    # No display
    return "appsrc name=app_source is-live=true leaky-type=upstream max-buffers=1 ! video/x-raw, format=RGB, width=2304, height=1296 !  queue name=source_scale_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoscale name=source_videoscale n-threads=2 ! queue name=source_convert_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoconvert n-threads=3 name=source_convert qos=false ! video/x-raw, pixel-aspect-ratio=1/1, format=RGB, width=2304, height=1296  ! queue name=inference_wrapper_input_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailocropper name=inference_wrapper_crop so-path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true hailoaggregator name=inference_wrapper_agg inference_wrapper_crop. ! queue name=inference_wrapper_bypass_q leaky=downstream max-size-buffers=10 max-size-bytes=0 max-size-time=0  ! inference_wrapper_agg.sink_0 inference_wrapper_crop. ! queue name=inference_scale_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoscale name=inference_videoscale n-threads=2 qos=false ! queue name=inference_convert_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! video/x-raw, pixel-aspect-ratio=1/1 ! videoconvert name=inference_videoconvert n-threads=2 ! queue name=inference_hailonet_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailonet name=inference_hailonet hef-path=/home/sai/Robot/resources/yolov11n.hef batch-size=2  vdevice-group-id=1 nms-score-threshold=0.3 nms-iou-threshold=0.45 output-format-type=HAILO_FORMAT_TYPE_FLOAT32 force-writable=true  ! queue name=inference_hailofilter_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailofilter name=inference_hailofilter so-path=/home/sai/Robot/venv_hailo_rpi5_examples/lib/python3.11/site-packages/hailo_apps_infra/../resources/libyolo_hailortpp_postprocess.so   function-name=filter_letterbox  qos=false ! queue name=inference_output_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0   ! inference_wrapper_agg.sink_1 inference_wrapper_agg. ! queue name=inference_wrapper_output_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0   ! queue name=identity_callback_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! identity name=identity_callback ! fakesink"

def camera_callback(pad, info):
    print("Camera callback called")
    return Gst.PadProbeReturn.OK
    start_time = time.time()

    # Get the GstBuffer from the probe info
    buffer: Gst.Buffer
    buffer = info.get_buffer()

    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK

    cameraside = None
    if buffer.has_flags(Gst.BufferFlags(1)):
        cameraside = CameraSide.LEFT
    elif buffer.has_flags(Gst.BufferFlags(2)):
        cameraside = CameraSide.RIGHT
    else:
        raise Exception("Unknown camera source")

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # Get frame
    frame = None
    if format is not None and width is not None and height is not None:
        # Convert buffer to numpy array
        frame = get_numpy_from_buffer(buffer, format, width, height)

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get the detections from the buffer
        roi = hailo.get_roi_from_buffer(buffer)
        hailo_detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        hailo_detections: list[hailo.HAILO_DETECTION]

        processed_detections: List[Detection] = list()

        # Parse the detections
        for detection in hailo_detections:
            bbox = detection.get_bbox()
            confidence = detection.get_confidence()
            masks = detection.get_objects_typed(hailo.HAILO_CONF_CLASS_MASK)
            bbox_min = (bbox.xmin() * 2304, bbox.ymin() * 1296)
            processed_detection = Detection(
                frame=None,
                bbox_min=bbox_min,
                width=bbox.width() * 2304, 
                height=bbox.height() * 1296, 
                timestamp=buffer.pts, 
                confidence=confidence, 
                side=cameraside,
                time=datetime.datetime.now())
            processed_detections.append(processed_detection)
    
        if processed_detections:
            output_stream.put(processed_detections)
            # print(f"Processed {len(processed_detections)} detections in {time.time() - start_time:.2f} seconds")
        else:
            print("No detections found")

    return Gst.PadProbeReturn.OK




# -----------------------------------------------------------------------------------------------
# GStreamerApp class
# -----------------------------------------------------------------------------------------------
class App:
    def __init__(self):
        # Set the process title
        setproctitle.setproctitle("Hailo Python App")

        # Set up signal handler for SIGINT (Ctrl-C)
        signal.signal(signal.SIGINT, self.shutdown)

        # Initialize variables
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
            format='%(levelname)s: %(message)s',
            filemode='w'
        )

        self.pipeline_latency = 0
        self.batch_size = 2
        self.video_width = 2304
        self.video_height = 1296
        self.video_format = "RGB888"
        self.threads = []
        self.create_pipeline()

        self.error_occurred = False


    def on_fps_measurement(self, sink, fps, droprate, avgfps):
        # print(f"FPS: {fps:.2f}, Droprate: {droprate:.2f}, Avg FPS: {avgfps:.2f}")
        return True

    def create_pipeline(self):
        # Initialize GStreamer
        Gst.init(None)

        pipeline_string = self.get_pipeline_string()
        try:
            self.pipeline = Gst.parse_launch(pipeline_string)
            print("psred and launched")
        except Exception as e:
            print(e)
            print(pipeline_string)
            sys.exit(1)

        # Create a GLib Main Loop
        self.loop = GLib.MainLoop()

    def bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End-of-stream")
            self.on_eos()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, {debug}")
            self.shutdown()
        # QOS
        elif t == Gst.MessageType.QOS:
            # Handle QoS message here
            qos_element = message.src.get_name()
            print(f"QoS message received from {qos_element}")
        return True


    def on_eos(self):
        self.shutdown()
    
    def monitor_pipeline_status(self):
        """Query and print the current pipeline state and statistics"""
        state_return, state, pending = self.pipeline.get_state(0)
        
        # Convert state to string representation
        state_str = "UNKNOWN"
        if state == Gst.State.PLAYING:
            state_str = "PLAYING"
        elif state == Gst.State.PAUSED:
            state_str = "PAUSED"
        elif state == Gst.State.READY:
            state_str = "READY"
        elif state == Gst.State.NULL:
            state_str = "NULL"
        
        print("\n--- Pipeline Status Report ---")
        print(f"Pipeline state: {state_str}")
        
        # Check specific elements
        elements_to_check = ["app_source"]  # Add other relevant element names
        for element_name in elements_to_check:
            element = self.pipeline.get_by_name(element_name)
            if element:
                _, elem_state, _ = element.get_state(0)
                elem_state_str = "UNKNOWN"
                if elem_state == Gst.State.PLAYING:
                    elem_state_str = "PLAYING"
                elif elem_state == Gst.State.PAUSED:
                    elem_state_str = "PAUSED"
                elif elem_state == Gst.State.READY:
                    elem_state_str = "READY"
                elif elem_state == Gst.State.NULL:
                    elem_state_str = "NULL"
                print(f"  {element_name} state: {elem_state_str}")
        
        print("----------------------------")
        return True  # Keep the timer going


    def shutdown(self, signum=None, frame=None):
        print("Shutting down... Hit Ctrl-C again to force quit.")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self.pipeline.set_state(Gst.State.PAUSED)
        GLib.usleep(100000)  # 0.1 second delay

        self.pipeline.set_state(Gst.State.READY)
        GLib.usleep(100000)  # 0.1 second delay

        self.pipeline.set_state(Gst.State.NULL)
        GLib.idle_add(self.loop.quit)

    def get_pipeline_string(self):
        """Creates pipeline via string."""
        return PIPELINE_STRING(display=False)

    def run(self):
        # Add a watch for messages on the pipeline's bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)

        # Connect pad probe to the identity element
        identity_1 = self.pipeline.get_by_name("identity_callback")
        if identity_1 is None:
            print("Warning: identity_callback element not found, add <identity name=identity_callback> in your pipeline where you want the callback to be called.")
        else:
            identity_1_pad = identity_1.get_static_pad("src")
            identity_1_pad.add_probe(Gst.PadProbeType.BUFFER, camera_callback)
            print("dit the thing")

        # Disable QoS to prevent frame drops
        disable_qos(self.pipeline)

        picam_thread = threading.Thread(target=picamera_thread, args=[self.pipeline])
        self.threads.append(picam_thread)
        picam_thread.start()


        # Set the pipeline to PAUSED to ensure elements are initialized
        self.pipeline.set_state(Gst.State.PAUSED)

        # Set pipeline latency
        new_latency = self.pipeline_latency * Gst.MSECOND  # Convert milliseconds to nanoseconds
        self.pipeline.set_latency(new_latency)

        # Set pipeline to PLAYING state
        self.pipeline.set_state(Gst.State.PLAYING)
        print("Playing")

        # Run the GLib event loop
        self.loop.run()

        # Clean up
        try:
            print("Cleaning up...")
            self.pipeline.set_state(Gst.State.NULL)
            for t in self.threads:
                t.join()
        except Exception as e:
            print(f"Error during cleanup: {e}", file=sys.stderr)
        finally:
            if self.error_occurred:
                print("Exiting with error...", file=sys.stderr)
                sys.exit(1)
            else:
                print("Exiting...")
                sys.exit(0)

def picamera_thread(pipeline):
    appsrc = pipeline.get_by_name("app_source")
    appsrc.set_property("is-live", True)
    appsrc.set_property("format", Gst.Format.TIME)

    image_width = 2304 # Assuming this from your class App init, adjust if needed
    image_height = 1296 # Assuming this from your class App init, adjust if needed
    image_format = "RGB888" # Assuming this from your class App init, adjust if needed

    # Path to images - adjust base path as necessary
    base_image_path = "data/basketball_images"
    left_image_files = sorted(glob.glob(os.path.join(base_image_path, "left", "*.jpg"))) # Assuming jpg, adjust if other format
    right_image_files = sorted(glob.glob(os.path.join(base_image_path, "right", "*.jpg")))

    if not left_image_files or not right_image_files:
        print(f"No image files found in {base_image_path}/left or {base_image_path}/right. Check paths and file types.")
        appsrc.end_of_stream() # Signal EOS if no files
        return
    
    if len(left_image_files) != len(right_image_files):
        print("Warning: Mismatch in number of left and right images. Processing shorter list.")
        # Process only the number of pairs available
        num_pairs = min(len(left_image_files), len(right_image_files))
        left_image_files = left_image_files[:num_pairs]
        right_image_files = right_image_files[:num_pairs]


    appsrc.set_property(
        "caps",
        Gst.Caps.from_string(
            f"video/x-raw, format={image_format}, width={image_width}, height={image_height}, "
            f"framerate=10/1, pixel-aspect-ratio=1/1" # Assuming 10 FPS, adjust if needed
        )
    )
    frame_count = 0
    fps = 10 # Frames per second for timestamp calculation

    for left_img_path, right_img_path in zip(left_image_files, right_image_files):
        # Process left image
        left_image = cv2.imread(left_img_path)
        if left_image is None:
            logging.error(f"Failed to read left image: {left_img_path}")
            continue
        
        # Ensure image is in the correct format (e.g., RGB) and dimensions if necessary
        # Example: if images are BGR (OpenCV default) and pipeline expects RGB
        left_image_rgb = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
        # Example: resize if necessary
        if left_image_rgb.shape[1] != image_width or left_image_rgb.shape[0] != image_height:
            left_image_rgb = cv2.resize(left_image_rgb, (image_width, image_height))
        
        left_data = left_image_rgb.tobytes()
        left_time = time.perf_counter() # Approximate capture time

        buffer_left = Gst.Buffer.new_wrapped(left_data)
        buffer_left.pts = Gst.util_uint64_scale_int(frame_count, Gst.SECOND, fps)
        buffer_left.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, fps)
        buffer_left.set_flags(Gst.BufferFlags(1))  # Flag as left camera
        ret = appsrc.emit('push-buffer', buffer_left)
        if ret != Gst.FlowReturn.OK:
            print(f"Failed to push left buffer for {left_img_path}: {ret}")
            break
        logging.info(f"Pushed left frame for {os.path.basename(left_img_path)} PTS: {buffer_left.pts} at {left_time}")
        frame_count += 1

        # Process right image
        right_image = cv2.imread(right_img_path)
        if right_image is None:
            logging.error(f"Failed to read right image: {right_img_path}")
            continue

        right_image_rgb = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
        if right_image_rgb.shape[1] != image_width or right_image_rgb.shape[0] != image_height:
            right_image_rgb = cv2.resize(right_image_rgb, (image_width, image_height))
        
        right_data = right_image_rgb.tobytes()
        right_time = time.perf_counter()

        buffer_right = Gst.Buffer.new_wrapped(right_data)
        buffer_right.pts = Gst.util_uint64_scale_int(frame_count, Gst.SECOND, fps) # Use current frame_count for right image
        buffer_right.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, fps)
        buffer_right.set_flags(Gst.BufferFlags(2))  # Flag as right camera
        ret = appsrc.emit('push-buffer', buffer_right)
        if ret != Gst.FlowReturn.OK:
            print(f"Failed to push right buffer for {right_img_path}: {ret}")
            break
        logging.info(f"Pushed right frame for {os.path.basename(right_img_path)} PTS: {buffer_right.pts} at {right_time}")
        frame_count += 1
        
        # Simulate delay if needed, e.g., to match a certain framerate for testing
        time.sleep(1/fps) 

    # After loop finishes or breaks, signal End Of Stream
    print("All images processed, sending EOS to appsrc.")
    # appsrc.end_of_stream()


if __name__ == "__main__":
    app = App()
    app.run()