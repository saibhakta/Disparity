import cv2
import time
import os
import argparse
import numpy as np

# Attempt to import Picamera2 for Raspberry Pi
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("Critical: Picamera2 library not found. This script requires a Raspberry Pi with Picamera2.")
    # exit(1) # Consider exiting if Picamera2 is essential and not found


def capture_calibration_stereo_images(
    output_dir_base="data/calibration_images",
    left_cam_idx=1,
    right_cam_idx=0,
    width=2304,
    height=1296,
    framerate=10,
    exposure=8000,
    gain=10.0,
    preview_scale_percent=25
):
    if not PICAMERA2_AVAILABLE:
        print("Error: Picamera2 library is not available. Cannot proceed with camera capture.")
        return

    left_dir = os.path.join(output_dir_base, "left")
    right_dir = os.path.join(output_dir_base, "right")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    print(f"Initializing Left Camera (index {left_cam_idx}) and Right Camera (index {right_cam_idx})")
    try:
        picam2_left = Picamera2(left_cam_idx)
        picam2_right = Picamera2(right_cam_idx)
    except Exception as e:
        print(f"Error initializing Picamera2 instances: {e}")
        print("Please ensure cameras are connected and indices are correct.")
        if Picamera2.global_camera_info:
             print("Available cameras:", Picamera2.global_camera_info())
        return

    controls = {'FrameRate': float(framerate), 'ExposureTime': int(exposure), 'AnalogueGain': float(gain)}
    capture_config = {"size": (int(width), int(height)), "format": "RGB888"}

    print("Configuring left camera...")
    config_left = picam2_left.create_still_configuration(main=capture_config, controls=controls)
    picam2_left.configure(config_left)

    print("Configuring right camera...")
    config_right = picam2_right.create_still_configuration(main=capture_config, controls=controls)
    picam2_right.configure(config_right)

    print("Starting cameras...")
    picam2_left.start()
    picam2_right.start()
    time.sleep(2) # Allow cameras to settle

    print(f"Cameras started. Outputting to {output_dir_base}")
    print("Press 'c' to capture an image pair.")
    print("Press 'n' to skip and try again (re-runs countdown).")
    print("Press 'q' to quit.")

    img_counter = 0
    try:
        while True:
            print("Prepare for capture...")
            for i in range(3, 0, -1):
                print(f"{i}...")
                time.sleep(1)
            
            # Capture images
            frame_left = picam2_left.capture_array()
            frame_right = picam2_right.capture_array()
            
            # Combine frames for display
            combined_frame = np.hstack((frame_left, frame_right))
            
            preview_width = int(combined_frame.shape[1] * preview_scale_percent / 100)
            preview_height = int(combined_frame.shape[0] * preview_scale_percent / 100)
            resized_image = cv2.resize(combined_frame, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
            
            # Rotate resized image (as per user's example)
            resized_image_rotated = cv2.rotate(resized_image, cv2.ROTATE_180)

            cv2.imshow("Stereo Calibration Preview (L | R) - Rotated", resized_image_rotated)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('c'):
                timestamp = int(time.time() * 1000)
                img_counter += 1
                
                # Use a counter in filename for easier sorting if preferred, or stick to pure timestamp
                # For now, using timestamp as per the example.
                # If counter is preferred: f"left_{img_counter:03d}.jpg"
                
                # left_filename = f"left_{img_counter:03d}.jpg"
                # right_filename = f"right_{img_counter:03d}.jpg"
                # Or, for numbered files:
                left_filename = f"left_{img_counter:03}.jpg"
                right_filename = f"right_{img_counter:03}.jpg"


                left_path = os.path.join(left_dir, left_filename)
                right_path = os.path.join(right_dir, right_filename)
            
                # Convert RGB to BGR for OpenCV imwrite
                cv2.imwrite(left_path, cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR))
                cv2.imwrite(right_path, cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR))
            
                print(f"Captured and saved: {left_path} and {right_path}")
            
            elif key == ord('n'):
                print("Skipping capture, preparing for next attempt.")
                cv2.destroyWindow("Stereo Calibration Preview (L | R) - Rotated") # Close current preview
                continue 
            
            elif key == ord('q'):
                print("Quitting capture.")
                break
            else:
                print(f"Unknown key: {chr(key)}. Press 'c', 'n', or 'q'.")

            # Important to destroy the window if we are looping to show a new one,
            # or after capture if we don't immediately show another.
            # If 'c' was pressed, the loop continues and will show a new window after countdown.
            cv2.destroyWindow("Stereo Calibration Preview (L | R) - Rotated")


    finally:
        print("Stopping cameras...")
        picam2_left.stop()
        picam2_right.stop()
        cv2.destroyAllWindows()
        print(f"Total pairs captured: {img_counter if 'img_counter' in locals() else 0}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture stereo image pairs for camera calibration.")
    parser.add_argument("--output_dir", type=str, default="data/calibration_images", 
                        help="Base directory to save 'left' and 'right' image subdirectories.")
    parser.add_argument("--left_cam_idx", type=int, default=1, help="Index of the left camera.")
    parser.add_argument("--right_cam_idx", type=int, default=0, help="Index of the right camera.")
    parser.add_argument("--width", type=int, default=2304, help="Capture width.")
    parser.add_argument("--height", type=int, default=1296, help="Capture height.")
    parser.add_argument("--framerate", type=float, default=10, help="Desired framerate for camera configuration.")
    parser.add_argument("--exposure", type=int, default=8000, help="Camera exposure time in microseconds.")
    parser.add_argument("--gain", type=float, default=10.0, help="Analogue gain for the camera.")
    parser.add_argument("--preview_scale", type=int, default=25, 
                        help="Percentage to scale the preview image (e.g., 25 for 25%%).")
    
    args = parser.parse_args()

    if not PICAMERA2_AVAILABLE:
        # The initial check already prints a message. We might not need to call the function at all.
        print("Exiting script as Picamera2 is not available or failed to import.")
    else:
        capture_calibration_stereo_images(
            output_dir_base=args.output_dir,
            left_cam_idx=args.left_cam_idx,
            right_cam_idx=args.right_cam_idx,
            width=args.width,
            height=args.height,
            framerate=args.framerate,
            exposure=args.exposure,
            gain=args.gain,
            preview_scale_percent=args.preview_scale
        ) 