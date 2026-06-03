import time
import cv2 
import numpy as np
import threading
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =====================================================================
# TUNABLE CONFIGURATION PARAMETERS (ADJUST FLIGHT CHARACTERS HERE)
# =====================================================================
class Config:
    # Drone Speed Limits & Multipliers
    MAX_SPEED = 40           # Maximum velocity cap for safe indoor tracking
    MANUAL_SPEED = 50        # Constant velocity vector when pressing W,S,A,D
    
    # Hand Gesture Detection Thresholds
    CURL_THRESHOLD = 0.05    # Structural distance threshold to verify a finger is curled
    INDEX_EXTEND_MIN = 0.08  # Minimum distance to verify the index finger is extended

# =====================================================================
# GLOBAL STATE CONTAINERS & ASYNC CALBACKS
# =====================================================================
latest_gesture_result = None

def gesture_callback(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_gesture_result
    latest_gesture_result = result

# =====================================================================
# GEOMETRIC VECTOR HELPER FUNCTIONS
# =====================================================================
def is_finger_curled(tip, mcp):
    """Returns True if a given finger tip is curled down close/below its knuckle."""
    return tip.y > mcp.y - Config.CURL_THRESHOLD

def detect_pointing_direction(landmarks):
    """
    Analyzes hand structure vectors. Returns a string description of pointing direction
    if the index finger is extended and all other fingers are curled into the palm.
    """
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    index_extended = (index_mcp.y - index_tip.y) > Config.INDEX_EXTEND_MIN or abs(index_tip.x - index_mcp.x) > Config.INDEX_EXTEND_MIN
    
    others_curled = (is_finger_curled(middle_tip, landmarks[9]) and 
                     is_finger_curled(ring_tip, landmarks[13]) and 
                     is_finger_curled(pinky_tip, landmarks[17]))
    
    if index_extended and others_curled:
        dx = index_tip.x - index_mcp.x
        dy = index_tip.y - index_mcp.y
        
        if abs(dx) > abs(dy):
            return "POINTING LEFT" if dx < -0.05 else "POINTING RIGHT" if dx > 0.05 else "NEUTRAL"
        else:
            return "POINTING UP" if dy < -0.05 else "POINTING DOWN" if dy > 0.05 else "NEUTRAL"
            
    return "NEUTRAL"

# =====================================================================
# SYSTEM INITIALIZATION (LOCAL WEBCAM BENCH)
# =====================================================================
print("=" * 60)
print("INITIALIZING WORKBENCH SIMULATION")
print("=" * 60)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

# ArUco Configuration completely copied from test.py
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50) # Get the predefined dictionary (use 50 because tradeoff variety for speed)
aruco_params = cv2.aruco.DetectorParameters() # Default parameters
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params) # Create the ArUco marker detector using the specified dictionary and parameters
MARKER_SIZE = .1 # Physical size of marker is 100 mm

# MediaPipe Setup
model_path = 'gesture_recognizer.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    result_callback=gesture_callback
)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Environment Simulated Metrics
current_mode = 3 
mode_names = {1: "MANUAL KEYBOARD", 2: "HAND GESTURE", 3: "ARUCO TRACKING"}
simulated_is_flying = False
action_in_progress = False

# Diagnostic Memory
last_printed_state = {"vector": [0, 0, 0, 0], "gesture": "NEUTRAL", "tag_seen": False}

def simulate_async_maneuver(command_name):
    global action_in_progress, simulated_is_flying
    action_in_progress = True
    print(f"\n[MANEUVER START] Executing blocking operation: {command_name.upper()}...")
    time.sleep(3.0) 
    simulated_is_flying = (command_name == "takeoff")
    print(f"[MANEUVER COMPLETED] System State: Flying={simulated_is_flying}")
    action_in_progress = False

def get_keyboard_command():
    global action_in_progress, simulated_is_flying
    lr, fb, ud, yv = 0, 0, 0, 0
    key = cv2.waitKey(1) & 0xFF
    
    if not action_in_progress:
        if key == ord('t') and not simulated_is_flying:
            threading.Thread(target=simulate_async_maneuver, args=("takeoff",), daemon=True).start()
        elif key == ord('l') and simulated_is_flying:
            threading.Thread(target=simulate_async_maneuver, args=("land",), daemon=True).start()
    
    if key == ord('w'): fb = Config.MANUAL_SPEED   
    elif key == ord('s'): fb = -Config.MANUAL_SPEED  
    elif key == ord('a'): lr = -Config.MANUAL_SPEED  
    elif key == ord('d'): lr = Config.MANUAL_SPEED   
    elif key == ord('r'): ud = Config.MANUAL_SPEED   
    elif key == ord('f'): ud = -Config.MANUAL_SPEED  
    elif key == ord('q'): yv = -Config.MANUAL_SPEED  
    elif key == ord('e'): yv = Config.MANUAL_SPEED   
    
    elif key == ord('1'): return 1, (0, 0, 0, 0), False
    elif key == ord('2'): return 2, (0, 0, 0, 0), False
    elif key == ord('3'): return 3, (0, 0, 0, 0), False
    elif key == 27 or key == ord('x'): return current_mode, (0, 0, 0, 0), True
        
    return current_mode, (lr, fb, ud, yv), False

# =====================================================================
# MAIN FRAME COMPUTATION WHIRLPOOL
# =====================================================================
try:
    while True:
        success, raw_frame = cap.read()
        if not success or raw_frame is None:
            continue
            
        #frame = cv2.flip(raw_frame, 1) 
        #frame = cv2.resize(frame, (960, 720))
        resized_frame = cv2.resize(raw_frame, (960, 720))
        frame = cv2.flip(resized_frame, 1) 

        h, w, _ = frame.shape
        
        lr, fb, ud, yv = 0, 0, 0, 0
        active_gesture = "NEUTRAL"
        tag_detected_this_frame = False
        
        new_mode, kb_vals, should_quit = get_keyboard_command()
        if should_quit: break
        if new_mode != current_mode:
            current_mode = new_mode
            print(f"\n[MODE SWITCH] Switched to pipeline channel: {mode_names[current_mode]}")
        

        if current_mode == 1:
            lr, fb, ud, yv = kb_vals

        # -----------------------------------------------------------------
        # MODULAR CHANNEL 2: HAND GESTURE CLASSIFIER
        # -----------------------------------------------------------------
        elif current_mode == 2:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            recognizer.recognize_async(mp_image, int(time.time() * 1000))
            
            if latest_gesture_result and latest_gesture_result.hand_landmarks:
                for hand_landmarks in latest_gesture_result.hand_landmarks:
                    for lm in hand_landmarks:
                        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)
                    
                    active_gesture = detect_pointing_direction(hand_landmarks)
                    
                    if active_gesture == "POINTING UP":     ud = 30
                    elif active_gesture == "POINTING DOWN":   ud = -30
                    elif active_gesture == "POINTING LEFT":   lr = -30
                    elif active_gesture == "POINTING RIGHT":  lr = 30


        # -----------------------------------------------------------------
        # MODULAR CHANNEL 3: ARUCO REPLACED COMPLETELY FROM TEST.PY
        # -----------------------------------------------------------------
        elif current_mode == 3:
            # Look for markers, notice: cv2.flip() will fail the detection
            corners, ids, rejected = detector.detectMarkers(resized_frame)
            if ids is not None:
                tag_detected_this_frame = True
                print(f"Detected tag ID: {ids}")
                cv2.aruco.drawDetectedMarkers(resized_frame, corners, ids)
                frame = cv2.flip(resized_frame, 1)

        # =====================================================================
        # DIAGNOSTIC CONSOLE MANAGEMENT & RENDERS
        # =====================================================================
        current_vector = [lr, fb, ud, yv]
        
        if (current_vector != last_printed_state["vector"] or 
            active_gesture != last_printed_state["gesture"] or 
            tag_detected_this_frame != last_printed_state["tag_seen"]):
            
            if simulated_is_flying and not action_in_progress:
                print(f"[TX ENGAGED] Vectors -> LR: {lr} | FB: {fb} | UD: {ud} | YW: {yv} [Context: Gesture={active_gesture}, TagSeen={tag_detected_this_frame}]")
            elif not simulated_is_flying:
                print(f"[TX SAFE-BLOCKED] Drone Landed. Target Vector Intent: {current_vector} | Gesture: {active_gesture}")
                
            last_printed_state = {"vector": current_vector, "gesture": active_gesture, "tag_seen": tag_detected_this_frame}

        # UI Overlays Draws
        cv2.putText(frame, f"MODE: {mode_names[current_mode]}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"STATUS: {'FLYING' if simulated_is_flying else 'LANDED'}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 120, 0), 2)
        
        if current_mode == 2:
            cv2.putText(frame, f"GESTURE: {active_gesture}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
        cv2.putText(frame, "1: Manual | 2: Gesture | 3: ArUco | T/L: Takeoff/Land | ESC: Exit", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
        if action_in_progress:
            cv2.putText(frame, "EXECUTING MANEUVER INTERRUPT...", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Tello Robotics Target Pipeline", frame)

finally:
    print("\nWiping system workspace contexts...")
    try: recognizer.close()
    except: pass
    cv2.destroyAllWindows()
    print("Clean environment exit achieved.")