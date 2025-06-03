#---------------------------------------------------------
# Pipeline helper functions
#---------------------------------------------------------
def SOURCE(src_file):
    """
    Returns source for camera 1 or 2
    """
    source = (
            f'appsrc name=app_source is-live=true leaky-type=upstream max-buffers=3 ! '
            # f'videoflip name=videoflip video-direction=horiz qos=false ! '
            f'video/x-raw, format=RGB, width=640, height=640, framerate=30/1, pixel-aspect-ratio=1/1 ! '
        )
    # source = f"""fdsrc ! image/jpeg, width=2304, height=1296, framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw, format=RGB, width=2304, height=1296, framerate=30/1, pixel-aspect-ratio=1/1 ! """
    return source

def PREPROCESS():
    """Adds crop to source"""
    return ("queue max-size-buffers=3 max-size-time=0 max-size-bytes=0 leaky=upstream ! "
        # f'videoscale n-threads=2 ! '
        # f'queue max-size-buffers=3 max-size-time=0 max-size-bytes=0 leaky=upstream ! '
        # f'videoconvert n-threads=3 qos=false ! '
        # f'video/x-raw, format=RGB, pixel-aspect-ratio=1/1 ! ')
    )
def COMBINE(source1, source2):
    """Returns combination of two streams"""
    return f"{source1}roundrobin.sink_0 \n{source2}roundrobin.sink_1 \nhailoroundrobin name=roundrobin mode=0 ! \n"

def INFERENCE():
    """A string representing the GStreamer pipeline for inference"""
    string = (
        "queue name=source_convert_q leaky=upstream max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! "
        # "videoconvert n-threads=3 name=source_convert qos=false ! video/x-raw, format=RGB, pixel-aspect-ratio=1/1 ! "
        # "videoscale name=inference_videoscale n-threads=2 qos=false ! "
        # "video/x-raw, pixel-aspect-ratio=1/1 ! videoconvert name=inference_videoconvert n-threads=2 ! "
        # "queue name=inference_hailonet_q leaky=upstream max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! "
        # "hailonet name=inference_hailonet hef-path=resources/yolov5n_seg_basketball.hef batch-size=1 force-writable=true ! "
        # "queue name=inference_hailofilter_q leaky=upstream max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! "
        # "hailofilter name=inference_hailofilter so-path=/home/sai/Robot/resources/libyolov5seg_postprocess.so "
        # "config-path=/home/sai/Robot/resources/yolov5n_seg_basketball.json function-name=yolov5seg qos=false ! "
        # "queue name=identity_callback_q leaky=upstream max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! "
    )
    return string

def SPLIT(pipeline, pipeline_2 = None):
    """Splits the source and applies pipeline to each"""
    split = "hailostreamrouter name=router src_0::input-streams=\"<sink_0>\" src_1::input-streams=\"<sink_1>\" "
    if pipeline_2 is None:
        split += "router.src_0 ! " + pipeline + "router.src_1 ! " + pipeline
    else:
        split += "router.src_0 ! " + pipeline + "router.src_1 ! " + pipeline_2
    return split

def COMPOSITE(num):
    return f"videoconvert ! queue name=source_convert_{num} ! video/x-raw,format=RGBA ! compositor.sink_{num} "

def DISPLAY():
    """
    Composites both into one
    """
    display_pipeline = (
        # f'queue ! '        
        # f'hailooverlay ! '
        # f'queue ! '    
        f'videoconvert n-threads=2 qos=false ! '
        f'queue ! '    
        f'fpsdisplaysink'
    )
    # return 'fakesink'
    return display_pipeline

def USER_CALLBACK(camera_num):
    """
    creates callback functions
    """
    if camera_num == 1:
        user_callback_pipeline = (
            f'queue ! '
            f'identity name=callback_{camera_num} ! '
        )
    else:
        user_callback_pipeline = (
            f'queue ! '
            f'identity name=callback_{camera_num} ! '
        )

    return user_callback_pipeline

# def PIPELINE_STRING():
#     return "appsrc name=app_source is-live=true leaky-type=downstream max-buffers=3 ! video/x-raw, format=RGB, width=2304, height=1296 !  queue name=source_scale_q leaky=no max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoscale name=source_videoscale n-threads=2 ! queue name=source_convert_q leaky=no max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoconvert n-threads=3 name=source_convert qos=false ! video/x-raw, pixel-aspect-ratio=1/1, format=RGB, width=640, height=640  ! queue name=inference_wrapper_input_q leaky=no max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailocropper name=inference_wrapper_crop so-path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true hailoaggregator name=inference_wrapper_agg inference_wrapper_crop. ! queue name=inference_wrapper_bypass_q leaky=no max-size-buffers=20 max-size-bytes=0 max-size-time=0  ! inference_wrapper_agg.sink_0 inference_wrapper_crop. ! queue name=inference_scale_q leaky=no max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoscale name=inference_videoscale n-threads=2 qos=false ! queue name=inference_convert_q leaky=no max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! video/x-raw, pixel-aspect-ratio=1/1 ! videoconvert name=inference_videoconvert n-threads=2 ! queue name=inference_hailonet_q leaky=no max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailonet name=inference_hailonet hef-path=/home/sai/Robot/resources/yolov5n_seg_basketball.hef batch-size=2  vdevice-group-id=1  force-writable=true  ! queue name=inference_hailofilter_q leaky=no max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailofilter name=inference_hailofilter so-path=/home/sai/Robot/resources/libyolov5seg_postprocess.so  config-path=/home/sai/Robot/resources/yolov5n_seg_basketball.json   function-name=filter_letterbox  qos=false ! queue name=inference_output_q leaky=no max-size-buffers=1 max-size-bytes=0 max-size-time=0   ! queue name=inference_wrapper_output_q leaky=no max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! identity name=identity_callback  ! queue name=hailo_display_overlay_q leaky=no max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailooverlay name=hailo_display_overlay  ! queue name=hailo_display_videoconvert_q leaky=no max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoconvert name=hailo_display_videoconvert n-threads=2 qos=false ! queue name=hailo_display_q leaky=no max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! fpsdisplaysink name=hailo_display video-sink=autovideosink sync=false text-overlay=False signal-fps-measurements=true"

def PIPELINE_STRING(display=True):
    if display:
        return "appsrc name=app_source is-live=true leaky-type=downstream max-buffers=1 ! video/x-raw, format=RGB, width=2304, height=1296 !  queue name=source_scale_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoscale name=source_videoscale n-threads=2 ! queue name=source_convert_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoconvert n-threads=3 name=source_convert qos=false ! video/x-raw, pixel-aspect-ratio=1/1, format=RGB, width=2304, height=1296  ! queue name=inference_wrapper_input_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailocropper name=inference_wrapper_crop so-path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true hailoaggregator name=inference_wrapper_agg inference_wrapper_crop. ! queue name=inference_wrapper_bypass_q leaky=downstream max-size-buffers=10 max-size-bytes=0 max-size-time=0  ! inference_wrapper_agg.sink_0 inference_wrapper_crop. ! queue name=inference_scale_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoscale name=inference_videoscale n-threads=2 qos=false ! queue name=inference_convert_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! video/x-raw, pixel-aspect-ratio=1/1 ! videoconvert name=inference_videoconvert n-threads=2 ! queue name=inference_hailonet_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailonet name=inference_hailonet hef-path=/home/sai/Robot/resources/yolov11n.hef batch-size=2  vdevice-group-id=1 nms-score-threshold=0.3 nms-iou-threshold=0.45 output-format-type=HAILO_FORMAT_TYPE_FLOAT32 force-writable=true  ! queue name=inference_hailofilter_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailofilter name=inference_hailofilter so-path=/home/sai/Robot/venv_hailo_rpi5_examples/lib/python3.11/site-packages/hailo_apps_infra/../resources/libyolo_hailortpp_postprocess.so   function-name=filter_letterbox  qos=false ! queue name=inference_output_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0   ! inference_wrapper_agg.sink_1 inference_wrapper_agg. ! queue name=inference_wrapper_output_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0   ! queue name=identity_callback_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! identity name=identity_callback ! queue name=hailo_display_overlay_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailooverlay name=hailo_display_overlay  ! queue name=hailo_display_videoconvert_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoconvert name=hailo_display_videoconvert n-threads=2 qos=false ! queue name=hailo_display_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! fpsdisplaysink name=hailo_display video-sink=autovideosink sync=false text-overlay=False signal-fps-measurements=true"
    
    # No display
    return "appsrc name=app_source is-live=true leaky-type=upstream max-buffers=1 ! video/x-raw, format=RGB, width=2304, height=1296 !  queue name=source_scale_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoscale name=source_videoscale n-threads=2 ! queue name=source_convert_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoconvert n-threads=3 name=source_convert qos=false ! video/x-raw, pixel-aspect-ratio=1/1, format=RGB, width=2304, height=1296  ! queue name=inference_wrapper_input_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailocropper name=inference_wrapper_crop so-path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true hailoaggregator name=inference_wrapper_agg inference_wrapper_crop. ! queue name=inference_wrapper_bypass_q leaky=downstream max-size-buffers=10 max-size-bytes=0 max-size-time=0  ! inference_wrapper_agg.sink_0 inference_wrapper_crop. ! queue name=inference_scale_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! videoscale name=inference_videoscale n-threads=2 qos=false ! queue name=inference_convert_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! video/x-raw, pixel-aspect-ratio=1/1 ! videoconvert name=inference_videoconvert n-threads=2 ! queue name=inference_hailonet_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailonet name=inference_hailonet hef-path=/home/sai/Robot/resources/yolov11n.hef batch-size=2  vdevice-group-id=1 nms-score-threshold=0.3 nms-iou-threshold=0.45 output-format-type=HAILO_FORMAT_TYPE_FLOAT32 force-writable=true  ! queue name=inference_hailofilter_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! hailofilter name=inference_hailofilter so-path=/home/sai/Robot/venv_hailo_rpi5_examples/lib/python3.11/site-packages/hailo_apps_infra/../resources/libyolo_hailortpp_postprocess.so   function-name=filter_letterbox  qos=false ! queue name=inference_output_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0   ! inference_wrapper_agg.sink_1 inference_wrapper_agg. ! queue name=inference_wrapper_output_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0   ! queue name=identity_callback_q leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0  ! identity name=identity_callback ! fakesink"


# In pipeline_helpers.py
def PIPELINE_STRING_FAST_DETECTION():
    return (
        "appsrc name=app_source is-live=true leaky-type=upstream max-buffers=2 ! " # Max-buffers 2 for a little slack
        "video/x-raw, format=RGB, width=2304, height=1296, framerate=10/1 ! "
        # Potentially a queue here if appsrc and hailocropper are in different clock domains or have jitter
        "queue name=pre_crop_q leaky=downstream max-size-buffers=2 ! "
        "hailocropper name=inference_wrapper_crop so-path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true ! "
        # Assuming hailocropper outputs what hailonet needs, or add minimal videoconvert/videoscale if necessary
        "queue name=pre_infer_q leaky=downstream max-size-buffers=2 ! "
        "hailonet name=inference_hailonet hef-path=/home/sai/Robot/resources/yolov11n.hef batch-size=2 nms-score-threshold=0.3 nms-iou-threshold=0.45 output-format-type=HAILO_FORMAT_TYPE_FLOAT32 force-writable=true ! "
        "queue name=pre_postproc_q leaky=downstream max-size-buffers=2 ! "
        "hailofilter name=inference_hailofilter so-path=/path/to/libyolo_hailortpp_postprocess.so function-name=filter_letterbox qos=false ! "
        # Callback directly after post-processing
        "identity name=identity_callback ! "
        "fakesink sync=false async=false" # Ensure fakesink doesn't block
    )