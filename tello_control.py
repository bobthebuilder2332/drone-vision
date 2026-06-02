import sys
print("CURRENT PYTHON EXECUTABLE PATH IS:", sys.executable)
import cv2 
import numpy as np

# FIX: Explicitly import the solutions submodule directly
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_draw
from djitellopy import Tello

# Initialize Tello Drone
drone = Tello()
drone.connect()
drone.streamon()

# Print battery status immediately for safety
print(f"Battery Level: {drone.get_battery()}%")

# Computer Vision Initialization
# Change these lines:
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils

# To this:
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
# (mp_draw is already imported directly above, use it as is)

# ArUco Tag Configuration (4x4, 50 variants)
# For OpenCV 4.7.x and newer, cv2.aruco.Dictionary_get is deprecated. Use getPredefinedDictionary.
try:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    legacy_aruco = False
except AttributeError:
    # Fallback for older OpenCV versions
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters_create()
    legacy_aruco = True

# Control Modes
# Modes: 0 = Manual Keyboard, 1 = Gesture Control, 2 = ArUco Tracking
current_mode = 0 
mode_names = {0: "MANUAL KEYBOARD", 1: "HAND GESTURE", 2: "ARUCO TRACKING"}

# Drone Camera Matrix (Approximated for Tello 720p stream)
# Essential for ArUco 3D distance calculations
focal_length = 920.0  
center_x, center_y = 480, 360
camera_matrix = np.array([[focal_length, 0, center_x],
                          [0, focal_length, center_y],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))  # Assuming minimal distortion

# Target distance from ArUco tag in cm
TARGET_DISTANCE = 50.0 

def get_keyboard_command():
    """
    Polls standard keyboard states via OpenCV's waitKey.
    Returns: left_right, forward_backward, up_down, yaw
    """
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50
    
    key = cv2.waitKey(1) & 0xFF
    
    # Flight State Commands
    if key == ord('t'):
        drone.takeoff()
    elif key == ord('l'):
        drone.land()
    
    # Movement Commands
    elif key == ord('w'): fb = speed   # Forward
    elif key == ord('s'): fb = -speed  # Backward
    elif key == ord('a'): lr = -speed  # Left
    elif key == ord('d'): lr = speed   # Right
    elif key == ord('r'): ud = speed   # Up
    elif key == ord('f'): ud = -speed  # Down
    elif key == ord('q'): yv = -speed  # Yaw Left
    elif key == ord('e'): yv = speed   # Yaw Right
    
    # Mode Switching
    elif key == ord('0'): return 0, (0, 0, 0, 0)
    elif key == ord('1'): return 1, (0, 0, 0, 0)
    elif key == ord('2'): return 2, (0, 0, 0, 0)
    
    return current_mode, (lr, fb, ud, yv)

try:
    while True:
        # 1. Fetch frame from drone
        frame_read = drone.get_frame_read()
        frame = frame_read.frame
        if frame is None:
            continue
            
        frame = cv2.resize(frame, (960, 720))
        h, w, _ = frame.shape
        
        # Initialize zero velocity components for this frame iteration
        lr, fb, ud, yv = 0, 0, 0, 0
        
        # Check keyboard first for overrides and mode toggles
        new_mode, kb_vals = get_keyboard_command()
        if new_mode != current_mode:
            current_mode = new_mode
            print(f"Switched to Mode: {mode_names[current_mode]}")
            
        if current_mode == 0:
            # Manual execution directly takes keyboard values
            lr, fb, ud, yv = kb_vals

        # 2. Automation Loop: Hand Gesture Control (Mode 1)
        elif current_mode == 1:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Basic gesture rule: Use Wrist (0) and Middle Finger Tip (12)
                    wrist = hand_landmarks.landmark[0]
                    middle_tip = hand_landmarks.landmark[12]
                    
                    # Target bounding-box error computation
                    if middle_tip.y < wrist.y - 0.2:  # Hand raised high
                        fb = 30  # Move Forward
                    elif middle_tip.y > wrist.y - 0.05: # Hand low
                        fb = -30 # Move Backward
                    
                    if middle_tip.x < wrist.x - 0.1:
                        lr = -30 # Move Left
                    elif middle_tip.x > wrist.x + 0.1:
                        lr = 30  # Move Right

        # 3. Automation Loop: ArUco Tracking (Mode 2)
        elif current_mode == 2:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if legacy_aruco:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
            else:
                corners, ids, _, _ = aruco_detector.detectMarkers(gray)
                
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                
                # Use SolvePnP via OpenCV ArUco module to estimate pose
                # markerLength parameter is set to 0.10 meters (10 cm) as specified
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.10, camera_matrix, dist_coeffs)
                
                if rvecs is not None and len(rvecs) > 0:
                    # Draw coordinate axes for visual validation
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], 0.05)
                    
                    # Extract translation vector components (in meters converted to cm)
                    x_err = tvecs[0][0][0] * 100 
                    y_err = tvecs[0][0][1] * 100
                    z_dist = tvecs[0][0][2] * 100
                    
                    # Proportional Tracking Control Loop
                    lr = int(np.clip(x_err * 1.5, -40, 40))   # Correct left/right offset
                    ud = int(np.clip(-y_err * 1.5, -40, 40))  # Correct height offset (invert Y)
                    
                    # Keep drone at target distance (50cm) from tag
                    dist_err = z_dist - TARGET_DISTANCE
                    if abs(dist_err) > 5:
                        fb = int(np.clip(dist_err * 1.2, -35, 35))

        # 4. Push Commands and Telemetry Overlay
        # Send raw RC velocities to drone channels safely
        if drone.is_flying:
            drone.send_rc_control(lr, fb, ud, yv)
            
        # Draw UI Overlay on frame
        cv2.putText(frame, f"MODE: {mode_names[current_mode]}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "0: Manual | 1: Gesture | 2: ArUco", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "T: Takeoff | L: Land | W/S/A/D: Move", (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("RoboMaster TT Live Feed", frame)
        
        # Stop processing if 'Esc' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # Graceful shutdown pipeline
    drone.send_rc_control(0, 0, 0, 0) # Neutralize drift
    drone.streamoff()
    cv2.destroyAllWindows()