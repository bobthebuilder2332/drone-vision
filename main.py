import time
import cv2
import threading
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from djitellopy import Tello 



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
    
    #don't forget the abs here
    index_extended = abs(index_mcp.y - index_tip.y) > Config.INDEX_EXTEND_MIN or abs(index_tip.x - index_mcp.x) > Config.INDEX_EXTEND_MIN
    
    others_curled = (is_finger_curled(middle_tip, landmarks[9]) and 
                     is_finger_curled(ring_tip, landmarks[13]) and 
                     is_finger_curled(pinky_tip, landmarks[17]))
    print("Only Index Extended", index_extended, others_curled)
    if index_extended and others_curled:
        dx = index_tip.x - index_mcp.x
        dy = index_tip.y - index_mcp.y
        
        if abs(dx) > abs(dy):
            return "POINTING LEFT" if dx < -0.05 else "POINTING RIGHT" if dx > 0.05 else "NEUTRAL"
        else:
            return "POINTING UP" if dy < -0.05 else "POINTING DOWN" if dy > 0.05 else "NEUTRAL"
            
    return "NEUTRAL"


# =====================================================================
# SYSTEM INITIALIZATION
# =====================================================================
print("=" * 60)
print("INITIALIZING TELLO ROBOTICS PIPELINE")
print("=" * 60)

drone = Tello()
drone.connect()
drone.streamon()
print("\n\n\n")
print(f"Battery Level: {drone.get_battery()}%")
print("\n\n\n")
drone.send_rc_control(0,0,0,0) # reset saved command carry over from last run

# ArUco Configuration 
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50) 
aruco_params = cv2.aruco.DetectorParameters() 
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params) 
MARKER_SIZE = .1 

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

# Environment Metrics
current_mode = 1
mode_names = {1: "MANUAL KEYBOARD", 2: "HAND GESTURE", 3: "ARUCO TRACKING", 4: "AUTO PILOT", 5: "AUTO LANDING"}
action_in_progress = False

# Diagnostic Memory
last_printed_state = {"vector": [0, 0, 0, 0], "gesture": "NEUTRAL", "tag_seen": False}

def run_async_maneuver(target_function):
    """Executes a blocking drone takeoff/landing maneuver inside an isolated thread."""
    global action_in_progress
    action_in_progress = True
    try:
        target_function()
    except Exception as err:
        print(f"[MANEUVER ERROR] Failed command: {err}")
    finally:
        action_in_progress = False

def turnMove():
    global current_mode
    drone.rotate_clockwise(180)
    drone.set_video_direction(drone.CAMERA_DOWNWARD)
    drone.move_forward(50)
    current_mode = 4  # Auto pilot

def get_keyboard_command():
    global action_in_progress
    lr, fb, ud, yv = 0, 0, 0, 0
    key = cv2.waitKey(1) & 0xFF
    
    if not action_in_progress:
        if key == ord('t') and not drone.is_flying:
            print("\n[FLIGHT COMMAND] Spawning Asynchronous Takeoff Thread...")
            threading.Thread(target=run_async_maneuver, args=(drone.takeoff,), daemon=True).start()
        elif key == ord('l') and drone.is_flying:
            print("\n[FLIGHT COMMAND] Spawning Asynchronous Landing Thread...")
            threading.Thread(target=run_async_maneuver, args=(drone.land,), daemon=True).start()
    
    if key == ord('w'): fb = Config.MANUAL_SPEED
    elif key == ord('s'): fb = -Config.MANUAL_SPEED
    elif key == ord('a'): lr = Config.MANUAL_SPEED
    elif key == ord('d'): lr = -Config.MANUAL_SPEED
    elif key == ord('r'): ud = Config.MANUAL_SPEED
    elif key == ord('f'): ud = -Config.MANUAL_SPEED
    elif key == ord('q'): yv = -Config.MANUAL_SPEED
    elif key == ord('e'): yv = Config.MANUAL_SPEED

    elif key == ord('b'): print(drone.get_battery())
    
    elif key == ord('1'):
        drone.set_video_direction(drone.CAMERA_FORWARD)
        return 1, (0, 0, 0, 0), False
    elif key == ord('2'):
        drone.set_video_direction(drone.CAMERA_FORWARD)
        return 2, (0, 0, 0, 0), False
    elif key == ord('3'):
        drone.set_video_direction(drone.CAMERA_FORWARD)
        return 3, (0, 0, 0, 0), False
    elif key == ord('4'):
        drone.set_video_direction(drone.CAMERA_DOWNWARD)
        return 4, (0, 0, 0, 0), False 
    elif key == ord('5'):
        drone.set_video_direction(drone.CAMERA_DOWNWARD)
        return 5, (0, 0, 0, 0), False
    elif key == 27 or key == ord('x'): return current_mode, (0, 0, 0, 0), True

    return current_mode, (lr, fb, ud, yv), False

def send_keepalive():
    while True:
        time.sleep(5)
        print("send keep alive")
        print(f"Battery Level: {drone.get_battery()}%")
        print("=" * 50)
threading.Thread(target=send_keepalive, daemon=True)

FRAME_WIDTH=960
FRAME_HEIGHT=720

# =====================================================================
# MAIN FRAME COMPUTATION WHIRLPOOL
# =====================================================================
try:
    while True:        
        frame_read = drone.get_frame_read()
        raw_frame = frame_read.frame
        if raw_frame is None:
            continue
            
        raw_frame_bgr = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)

        resized_frame = cv2.resize(raw_frame_bgr, (FRAME_WIDTH, FRAME_HEIGHT))
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
            #Slow down this, lagging might lost control
            time.sleep(0.05)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            recognizer.recognize_async(mp_image, int(time.time() * 1000))

            if latest_gesture_result and latest_gesture_result.hand_landmarks:
                for hand_landmarks in latest_gesture_result.hand_landmarks:
                    for lm in hand_landmarks:
                        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)
                    
                    active_gesture = detect_pointing_direction(hand_landmarks)
                    print("active_gesture is ", active_gesture)

                    if active_gesture == "POINTING UP":     ud = 30
                    elif active_gesture == "POINTING DOWN":   ud = -30
                    elif active_gesture == "POINTING LEFT":   lr = Config.MANUAL_SPEED #old value: -30
                    elif active_gesture == "POINTING RIGHT":  lr = -Config.MANUAL_SPEED #old value: 30

        # -----------------------------------------------------------------
        # MODULAR CHANNEL 3: ARUCO TARGET DETECTION
        # -----------------------------------------------------------------
        elif current_mode == 3:
            corners, ids, rejected = detector.detectMarkers(resized_frame)
            if ids is not None:
                print(ids)
                tag_detected_this_frame = True
                cv2.aruco.drawDetectedMarkers(resized_frame, corners, ids)
                frame = cv2.flip(resized_frame, 1)
                match ids:
                    # ... add more cases here for more tags if needed
                    case 5: pass
                    case 6:
                        current_mode = 4 # waitting thread finish then continue, reuse keyboard mode
                        print(f"\n[MODE SWITCH] Switched to auto-pilot mode")
                        #threading.Thread(target=turnMove, daemon=True).start()
                    case 7: pass
                    # .... add more cases here for more tags if needed

        elif current_mode == 4:
            corners, ids, rejected = detector.detectMarkers(resized_frame)
            if ids is not None:
                print(ids)
                cv2.aruco.drawDetectedMarkers(resized_frame, corners, ids)
                frame = cv2.flip(resized_frame, 1)
                match ids:
                    case 7:
                        current_mode = 5 # waitting thread finish then continue, reuse keyboard mode
                        print(f"\n[MODE SWITCH] Switched to auto-landing mode")
                       
            else:
                #fb = Config.MANUAL_SPEED
                drone.move_back(20)

        elif current_mode == 5:
            drone.send_rc_control(0,0,0,0)
            
            corners, ids, rejected = detector.detectMarkers(resized_frame)
            if ids is not None:
                marker_corners = corners[0][0]
        
                # Calculate the center (average of X's and average of Y's)
                center_x = int((marker_corners[0][0] + marker_corners[1][0] + marker_corners[2][0] + marker_corners[3][0]) / 4)
                center_y = int((marker_corners[0][1] + marker_corners[1][1] + marker_corners[2][1] + marker_corners[3][1]) / 4)
                
                # Output the coordinates
                print(f"Marker ID {ids} Center -> X: {center_x}, Y: {center_y}")
                
                # Optional: Draw a circle at the center for visual debugging
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                cv2.circle(frame, (int(FRAME_WIDTH/2), int(FRAME_HEIGHT/2)), 5, (255, 0, 0), -1)

                delta_x = center_x - int(FRAME_WIDTH/2)
                delta_y = center_y - int(FRAME_HEIGHT/2)

                print(f"Marker ID {ids} Delta -> X: {delta_x}, Y: {delta_y}")

                
                # For down camera, X - fb direction, Y - lr direction
                if delta_x > 40: 
                    drone.move_back(20)
                    #drone.send_rc_control(0, -20, 0, 0)
                    #time.sleep(0.1)
                    #drone.send_rc_control(0,0,0,0)
                elif delta_x < -40: 
                    drone.move_forward(20)
                    #drone.send_rc_control(0, 20, 0, 0)
                    #time.sleep(0.1)
                    #drone.send_rc_control(0,0,0,0)
                elif delta_y > 40: 
                    drone.move_left(20)
                    #drone.send_rc_control(-20, 0, -20, 0)
                    #time.sleep(0.1)
                    #drone.send_rc_control(0,0,0,0)
                elif delta_y < -40: 
                    drone.move_right(20)
                    #drone.send_rc_control(20, 0, -20, 0)
                    #time.sleep(0.1)
                    #drone.send_rc_control(0,0,0,0)
                else: 
                    print("Landing")
                    threading.Thread(target=run_async_maneuver, args=(drone.land,), daemon=True).start()
                    current_mode = 1




                
               

        # =====================================================================
        # FLIGHT CONTROL DISPATCH & DIAGNOSTICS
        # =====================================================================
        # Actually dispatch velocities to the drone if it is airborne and not executing a takeoff/land
        if drone.is_flying and not action_in_progress:
            if lr or fb or ud or yv:
                drone.send_rc_control(lr, fb, ud, yv)
                time.sleep(0.1)
                drone.send_rc_control(0,0,0,0)

        current_vector = [lr, fb, ud, yv]
        
        if (current_vector != last_printed_state["vector"] or 
            active_gesture != last_printed_state["gesture"] or 
            tag_detected_this_frame != last_printed_state["tag_seen"]):
            
            if drone.is_flying and not action_in_progress:
                print(f"[TX ENGAGED] Live Command -> LR: {lr} | FB: {fb} | UD: {ud} | YW: {yv} [Context: Gesture={active_gesture}, TagSeen={tag_detected_this_frame}]")
            elif not drone.is_flying:
                print(f"[TX BLOCK] Grounded. Safe-blocked Vector Intent: {current_vector}")
                
            last_printed_state = {"vector": current_vector, "gesture": active_gesture, "tag_seen": tag_detected_this_frame}

        # UI Overlays Draws
        cv2.putText(frame, f"MODE: {mode_names[current_mode]}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"STATUS: {'FLYING' if drone.is_flying else 'LANDED'}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 120, 0), 2)
        
        if current_mode == 2:
            cv2.putText(frame, f"GESTURE: {active_gesture}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
        cv2.putText(frame, "1: Manual | 2: Gesture | 3: ArUco | T/L: Takeoff/Land | ESC: Exit", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
        if action_in_progress:
            cv2.putText(frame, "EXECUTING HARDWARE MANEUVER...", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Tello Robotics Target Pipeline", frame)

finally:
    print("\nExecuting Safe System Shutdown Sequence...")
    
    try: 
        if drone.is_flying:
            threading.Thread(target=run_async_maneuver, args=(drone.land,), daemon=True).start()
        drone.send_rc_control(0, 0, 0, 0)
        drone.streamoff()  
    except Exception as shutdown_err: 
        print(f"Drone cleanup notice: {shutdown_err}")
    
    try: recognizer.close()
    except: pass
    cv2.destroyAllWindows()
    print("Clean environment exit achieved.")