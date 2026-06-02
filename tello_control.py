import sys
import time
import cv2 
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from djitellopy import Tello

# =====================================================================
# GLOBAL CONTAINER FOR ASYNCHRONOUS TASKS API LANDMARKS
# =====================================================================
latest_gesture_result = None

def gesture_callback(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    """Asynchronous callback triggered whenever MediaPipe finishes processing a frame."""
    global latest_gesture_result
    latest_gesture_result = result

# =====================================================================
# SYSTEM INITIALIZATION
# =====================================================================

# Initialize Tello Drone
drone = Tello()
drone.connect()
drone.streamon()
print(f"Battery Level: {drone.get_battery()}%")

# ArUco Tag Configuration (4x4, 50 variants)
try:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    legacy_aruco = False
except AttributeError:
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters_create()
    legacy_aruco = True

# MediaPipe Modern Tasks API Configuration
model_path = 'gesture_recognizer.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    result_callback=gesture_callback
)
recognizer = vision.GestureRecognizer.create_from_options(options)

# =====================================================================
# FLIGHT CONFIGURATION & STATE METRICS
# =====================================================================
# Updated Control Modes: 1 = Manual Keyboard, 2 = Gesture Control, 3 = ArUco Tracking
current_mode = 1 
mode_names = {1: "MANUAL KEYBOARD", 2: "HAND GESTURE", 3: "ARUCO TRACKING"}

focal_length = 920.0  
center_x, center_y = 480, 360
camera_matrix = np.array([[focal_length, 0, center_x],
                          [0, focal_length, center_y],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))

TARGET_DISTANCE = 50.0  # Target distance from ArUco tag in cm

def get_keyboard_command():
    """Polls standard keyboard states via OpenCV's waitKey."""
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50
    
    key = cv2.waitKey(1) & 0xFF
    
    # Flight State Commands
    if key == ord('t'):
        drone.takeoff()
    elif key == ord('l'):
        drone.land()
    
    # Movement Controls
    elif key == ord('w'): fb = speed   # Forward
    elif key == ord('s'): fb = -speed  # Backward
    elif key == ord('a'): lr = -speed  # Left
    elif key == ord('d'): lr = speed   # Right
    elif key == ord('r'): ud = speed   # Up
    elif key == ord('f'): ud = -speed  # Down
    elif key == ord('q'): yv = -speed  # Yaw Left
    elif key == ord('e'): yv = speed   # Yaw Right
    
    # Updated Mode Switching Inputs
    elif key == ord('1'): return 1, (0, 0, 0, 0), False
    elif key == ord('2'): return 2, (0, 0, 0, 0), False
    elif key == ord('3'): return 3, (0, 0, 0, 0), False
    
    # Explicit Exit/Quit Keys
    elif key == 27 or key == ord('x'): # 27 is the Escape key code
        return current_mode, (0, 0, 0, 0), True
        
    return current_mode, (lr, fb, ud, yv), False

# =====================================================================
# MAIN FLIGHT EXECUTION STREAM
# =====================================================================
try:
    while True:
        frame_read = drone.get_frame_read()
        raw_frame = frame_read.frame
        if raw_frame is None:
            continue
            
        # Convert Native Tello RGB array to BGR array for correct OpenCV colors
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (960, 720))
        h, w, _ = frame.shape
        
        # Reset zeroed base velocity vectors for this loop cycle
        lr, fb, ud, yv = 0, 0, 0, 0
        
        new_mode, kb_vals, should_quit = get_keyboard_command()
        
        # Handle explicit exit interrupt
        if should_quit:
            print("\nExit command detected. Initiating landing sequence...")
            break
            
        if new_mode != current_mode:
            current_mode = new_mode
            print(f"Switched to Mode: {mode_names[current_mode]}")
            
        if current_mode == 1:
            lr, fb, ud, yv = kb_vals

        # -----------------------------------------------------------------
        # MODE 2: MODERN GESTURE TASKS CONTROL
        # -----------------------------------------------------------------
        elif current_mode == 2:
            rgb_resized = cv2.resize(raw_frame, (960, 720))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_resized)
            
            timestamp_ms = int(time.time() * 1000)
            recognizer.recognize_async(mp_image, timestamp_ms)
            
            if latest_gesture_result and latest_gesture_result.hand_landmarks:
                for hand_landmarks in latest_gesture_result.hand_landmarks:
                    # RENDER OVERLAYS ONLY IN GESTURE MODE
                    for landmark in hand_landmarks:
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                    
                    wrist = hand_landmarks[0]
                    middle_tip = hand_landmarks[12]
                    
                    if middle_tip.y < wrist.y - 0.2:    fb = 30  
                    elif middle_tip.y > wrist.y - 0.05:  fb = -30 
                    
                    if middle_tip.x < wrist.x - 0.1:     lr = -30 
                    elif middle_tip.x > wrist.x + 0.1:   lr = 30  

        # -----------------------------------------------------------------
        # MODE 3: ARUCO TAG TARGET TRACKING
        # -----------------------------------------------------------------
        elif current_mode == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if legacy_aruco:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
            else:
                # Fixed unpacking layout matching OpenCV runtime shapes
                corners, ids, _ = aruco_detector.detectMarkers(gray)
                
            if ids is not None:
                # RENDER OVERLAYS ONLY IN ARUCO MODE
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.10, camera_matrix, dist_coeffs)
                
                if rvecs is not None and len(rvecs) > 0:
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], 0.05)
                    
                    x_err = tvecs[0][0][0] * 100 
                    y_err = tvecs[0][0][1] * 100
                    z_dist = tvecs[0][0][2] * 100
                    
                    lr = int(np.clip(x_err * 1.5, -40, 40))   
                    ud = int(np.clip(-y_err * 1.5, -40, 40))  
                    
                    dist_err = z_dist - TARGET_DISTANCE
                    if abs(dist_err) > 5:
                        fb = int(np.clip(dist_err * 1.2, -35, 35))
            else:
                lr, fb, ud, yv = 0, 0, 0, 0

        # =====================================================================
        # FLIGHT CONTROL DISPATCH & HUD TELEMETRY
        # =====================================================================
        if drone.is_flying:
            drone.send_rc_control(lr, fb, ud, yv)
            
        cv2.putText(frame, f"MODE: {mode_names[current_mode]}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "1: Manual | 2: Gesture | 3: ArUco", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "T: Takeoff | L: Land | ESC/X: Exit Script", (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Tello Robotics Target Pipeline", frame)

finally:
    print("\nExecuting Safe System Shutdown Sequence...")
    try:
        if drone.is_flying:
            drone.land()
        drone.send_rc_control(0, 0, 0, 0)
        drone.streamoff()
    except Exception as shutdown_err:
        print(f"Drone disconnect cleanup notice: {shutdown_err}")
        pass
    recognizer.close()
    cv2.destroyAllWindows()