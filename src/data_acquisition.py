import cv2
import time
import os
import argparse
import numpy as np

# Attempt to import Picamera2 for Raspberry Pi, but allow fallback for testing on other systems
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("Warning: Picamera2 library not found. Live capture from Pi Cameras will not be available.")
    print("You can still use this script with --use_dummy_cameras for testing image saving logic.")


def capture_stereo_images_picamera2(output_dir, num_pairs=10, delay_sec=2):
    if not PICAMERA2_AVAILABLE:
        print("Error: Picamera2 not available. Cannot capture from Raspberry Pi cameras.")
        return

    left_dir = os.path.join(output_dir, "left")
    right_dir = os.path.join(output_dir, "right")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    # Determine starting image index
    existing_files = os.listdir(left_dir)
    max_idx = 0
    for f in existing_files:
        if f.startswith("left_") and f.endswith(".jpg"):
            try:
                idx = int(f.replace("left_", "").replace(".jpg", ""))
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                pass # Ignore files not matching the pattern
    
    current_img_idx = max_idx + 1

    # Initialize cameras - assuming camera 0 is right, 1 is left (as in your original script)
    # You might need to adjust camera indices (0 or 1) based on your hardware setup
    try:
        cam_right_idx, cam_left_idx = 0, 1 # Default from your script
        print(f"Attempting to initialize Right Camera (index {cam_right_idx}) and Left Camera (index {cam_left_idx})")
        picam2_right = Picamera2(cam_right_idx)
        picam2_left = Picamera2(cam_left_idx)
    except Exception as e:
        print(f"Error initializing Picamera2 instances: {e}")
        print("Please ensure cameras are connected and indices are correct.")
        print("Available cameras:", Picamera2.global_camera_info())
        return
        
    # Configure cameras (example configuration, adjust as needed)
    # Using still configuration for higher quality captures if needed for calibration
    # For dynamic scenes like basketball, a video/preview config might be more representative
    # Your proposal uses IMX708 sensors
    # Resolution from your example: (2304, 1296)
    capture_config = {"size": (2304, 1296), "format": "RGB888"} # Or XRGB8888 if RGB888 gives issues
    
    # For calibration images, exposure control might be less critical if scene is static
    # For basketball, you might need to tune ExposureTime, AnalogueGain
    controls = {'FrameRate': 10, 'ExposureTime': 500, 'AnalogueGain': 2.0} 


    config_right = picam2_right.create_still_configuration(main=capture_config, controls=controls)
    config_left = picam2_left.create_still_configuration(main=capture_config, controls=controls)
    
    print("Configuring right camera...")
    picam2_right.configure(config_right)
    print("Configuring left camera...")
    picam2_left.configure(config_left)

    print("Starting cameras...")
    picam2_right.start()
    picam2_left.start()
    time.sleep(2) # Allow cameras to settle

    print(f"Starting capture of {num_pairs} stereo pairs.")
    print("A preview window will be shown (if desktop environment is available).")

    captured_count = 0
    for i in range(num_pairs):
        print(f"Waiting {delay_sec} seconds before capturing pair {i+1}...")
        time.sleep(delay_sec)
        print(f"Capturing pair {i+1} (Image index: {current_img_idx})...")

        # Capture high-resolution frames for saving
        # It's crucial these are captured as close in time as possible.
        # Picamera2's start() method for multiple cameras tries to sync them.
        frame_left = picam2_left.capture_array() # Captures from 'main' stream by default
        frame_right = picam2_right.capture_array()
        # timestamp = int(time.time() * 1000) # Use one timestamp for the pair - replaced by index
        # Small delay to ensure second capture completes if there's any slight offset
        # time.sleep(0.05)

        # Display the captured frames (resized for preview)
        preview_l_small = cv2.resize(frame_left, (0,0), fx=0.25, fy=0.25)
        preview_r_small = cv2.resize(frame_right, (0,0), fx=0.25, fy=0.25)
        combined_preview = np.hstack((preview_l_small, preview_r_small))
        cv2.imshow("Stereo Preview (L | R)", combined_preview)
        cv2.waitKey(1) # Allow GUI to update, non-blocking

        left_filename = f"left_{current_img_idx:02d}.jpg" # Using 2-digit padding, adjust if more needed
        right_filename = f"right_{current_img_idx:02d}.jpg"
        
        left_path = os.path.join(left_dir, left_filename)
        right_path = os.path.join(right_dir, right_filename)

        cv2.imwrite(left_path, frame_left) # OpenCV expects BGR
        cv2.imwrite(right_path, frame_right)

        print(f"Saved: {left_path} and {right_path}")
        captured_count += 1
        current_img_idx += 1 # Increment for next pair
        
        # if preview: # Close the specific preview window after capture if 'c' was pressed # Removed
        #      cv2.destroyWindow("Stereo Preview (L | R) - Press 'c' to capture, 'q' to quit") # Removed


    print(f"Captured {captured_count} pairs.")
    print("Stopping cameras...")
    picam2_left.stop()
    picam2_right.stop()
    cv2.destroyAllWindows() # Ensure all OpenCV windows are closed at the end

def capture_stereo_images_dummy(output_dir, num_pairs=5, delay_sec=1):
    print("Using DUMMY camera capture. Will save placeholder images.")
    left_dir = os.path.join(output_dir, "left")
    right_dir = os.path.join(output_dir, "right")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    img_height, img_width = 480, 640 # Dummy image size

    # Determine starting image index for dummy images
    existing_files = os.listdir(left_dir)
    max_idx = 0
    for f in existing_files:
        if f.startswith("left_") and f.endswith(".jpg"):
            try:
                idx = int(f.replace("left_", "").replace(".jpg", ""))
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                pass # Ignore files not matching the pattern
    current_dummy_idx = max_idx + 1


    for i in range(num_pairs):
        print(f"Capturing dummy pair {i+1}/{num_pairs} (Image index: {current_dummy_idx})...")
        # timestamp = int(time.time() * 1000) # Replaced by index

        dummy_img_left = np.random.randint(0, 256, (img_height, img_width, 3), dtype=np.uint8)
        # cv2.putText(dummy_img_left, f"Left Dummy {timestamp}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(dummy_img_left, f"Left Dummy {current_dummy_idx:02d}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        left_filename = f"left_{current_dummy_idx:02d}.jpg"
        left_path = os.path.join(left_dir, left_filename)
        cv2.imwrite(left_path, dummy_img_left)

        dummy_img_right = np.random.randint(0, 256, (img_height, img_width, 3), dtype=np.uint8)
        # cv2.putText(dummy_img_right, f"Right Dummy {timestamp}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(dummy_img_right, f"Right Dummy {current_dummy_idx:02d}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        right_filename = f"right_{current_dummy_idx:02d}.jpg"
        right_path = os.path.join(right_dir, right_filename)
        cv2.imwrite(right_path, dummy_img_right)

        print(f"Saved: {left_path} and {right_path}")
        current_dummy_idx +=1 # Increment for next pair
        if i < num_pairs - 1:
            time.sleep(delay_sec)
    print("Dummy capture complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture stereo image pairs.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save 'left' and 'right' image subdirectories.")
    parser.add_argument("--num_pairs", type=int, default=10, help="Number of stereo pairs to capture.")
    parser.add_argument("--delay_sec", type=int, default=2, help="Delay in seconds between automatic captures (if not using manual 'c' key).")
    parser.add_argument("--use_dummy_cameras", action='store_true', help="Use dummy camera capture for testing without Pi hardware.")
    args = parser.parse_args()

    if args.use_dummy_cameras:
        capture_stereo_images_dummy(args.output_dir, args.num_pairs, args.delay_sec)
    elif PICAMERA2_AVAILABLE:
        capture_stereo_images_picamera2(args.output_dir, args.num_pairs, args.delay_sec)
    else:
        print("Picamera2 library is not available and --use_dummy_cameras was not specified. Exiting.")