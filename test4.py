import cv2
import pygame
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Global container for the asynchronous worker thread results
latest_result = None
mode = 'tag'  # Set to 'gesture' if you want to skip needing ArUco Tag 4 to switch modes

def save_result(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# --- WEBCAM INITIALIZATION ---
cap = cv2.VideoCapture(1) 

# Simulated Tello telemetry metrics
battery_val = 99
battery = f'Battery: {battery_val}%'
print(f"{battery:=^40}")

# Initialize the ArUco marker detection
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Configurations for Gesture Recognizer Tasks API
model_path = 'gesture_recognizer.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.5, 
    min_hand_presence_confidence=0.5,
    result_callback=save_result
)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((960, 720))
pygame.display.set_caption(f"Webcam Drone Simulator Feed | {battery}")
clock = pygame.time.Clock()

# Drone control variables
speed = 50
rc_control = [0, 0, 0, 0]
last_rc_control = [0, 0, 0, 0]

# Main loop
with vision.GestureRecognizer.create_from_options(options) as recognizer:
    try:
        running = True
        frame_counter = 0
        while running:
            frame_counter += 1
            keyboard_active = False

            # Input handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    running = False

                if event.type == pygame.KEYDOWN:
                    match event.key:
                        case pygame.K_t: print("SIMULATED COMMAND: Takeoff")
                        case pygame.K_l: print("SIMULATED COMMAND: Land")
                        case pygame.K_e: print("SIMULATED COMMAND: EMERGENCY MOTOR CUT")
                        case pygame.K_q: running = False
                        case pygame.K_UP: rc_control[2] = speed
                        case pygame.K_DOWN: rc_control[2] = -speed
                        case pygame.K_LEFT: rc_control[3] = -speed
                        case pygame.K_RIGHT: rc_control[3] = speed
                        case pygame.K_w: rc_control[1] = speed
                        case pygame.K_s: rc_control[1] = -speed
                        case pygame.K_a: rc_control[0] = -speed
                        case pygame.K_d: rc_control[0] = speed
                        
                if event.type == pygame.KEYUP:
                    match event.key:
                        case pygame.K_UP | pygame.K_DOWN: rc_control[2] = 0
                        case pygame.K_LEFT | pygame.K_RIGHT: rc_control[3] = 0
                        case pygame.K_w | pygame.K_s: rc_control[1] = 0
                        case pygame.K_a | pygame.K_d: rc_control[0] = 0
            
            # Check if user is manually overriding drone via keys
            if rc_control != [0, 0, 0, 0]:
                keyboard_active = True

            # Fetch the capture device frame data
            ret, frame = cap.read()
            if ret and frame is not None:
                # Resize to standard Pygame display window surface bounds
                frame = cv2.resize(frame, (960, 720))
                
                # --- FIX: DETECT ARUCO BEFORE FLIPPING ---
                corners, ids, _ = detector.detectMarkers(frame)
                
                # Now safely mirror the frame for natural webcam user interaction
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                
                if ids is not None:
                    # Mirror the corner coordinates so the drawn boxes match the flipped preview
                    flipped_corners = []
                    for marker in corners:
                        marker_corners = marker[0]
                        flipped_marker = []
                        for corner in marker_corners:
                            flipped_x = w - corner[0]
                            flipped_y = corner[1]
                            flipped_marker.append([flipped_x, flipped_y])
                        flipped_corners.append([flipped_marker])
                    
                    cv2.aruco.drawDetectedMarkers(frame, flipped_corners, ids)
                    detected_ids = ids.flatten()
                else: 
                    detected_ids = []

                gesture_name = 'None'  
                fingers = 0

                if frame_counter % 150 == 0:
                    battery = f'Battery: {battery_val}%'
                    pygame.display.set_caption(f"Webcam Drone Simulator Feed | {battery}")

                # Convert video properties for MediaPipe tracking entry pipelines
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mediapipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                timestamp_ms = frame_counter * 33
                recognizer.recognize_async(mediapipe_image, timestamp_ms)

                if latest_result and latest_result.hand_landmarks:
                    for index, hand_landmarks in enumerate(latest_result.hand_landmarks):
                        for lm in hand_landmarks:
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                        
                        wrist = hand_landmarks[0]
                        index_base = hand_landmarks[5]
                        pinky_base = hand_landmarks[17]

                        hand_scale = math.sqrt((index_base.x - pinky_base.x)**2 + (index_base.y - pinky_base.y)**2)

                        thumb_tip = hand_landmarks[4]
                        thumb_ip = hand_landmarks[3]

                        tip_to_index_base = math.sqrt((thumb_tip.x - index_base.x)**2 + (thumb_tip.y - index_base.y)**2)
                        ip_to_index_base = math.sqrt((thumb_ip.x - index_base.x)**2 + (thumb_ip.y - index_base.y)**2)

                        if (tip_to_index_base / hand_scale) < 0.42 or (ip_to_index_base / hand_scale) < 0.38: 
                            thumb_open = 0
                        else: 
                            thumb_open = 1
                        
                        finger_tips = [8, 12, 16, 20]
                        finger_pips = [6, 10, 14, 18]
                        
                        fingers_open = [thumb_open]

                        for tip_id, pip_id in zip(finger_tips, finger_pips):
                            tip = hand_landmarks[tip_id]
                            pip = hand_landmarks[pip_id]
                            
                            dist_tip_to_wrist = math.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
                            dist_pip_to_wrist = math.sqrt((pip.x - wrist.x)**2 + (pip.y - wrist.y)**2)
                            
                            if dist_tip_to_wrist > dist_pip_to_wrist: 
                                fingers_open.append(1)
                            else: 
                                fingers_open.append(0)

                        fingers = sum(fingers_open)

                        # Classification rules engine structures
                        current_gesture = 'None'
                        if fingers_open == [0, 1, 0, 0, 0] or fingers_open == [1, 1, 0, 0, 0]:
                            idx_tip = hand_landmarks[8]
                            idx_base = hand_landmarks[5]

                            delta_x = idx_tip.x - idx_base.x
                            delta_y = idx_tip.y - idx_base.y
                            
                            if abs(delta_x) > abs(delta_y): 
                                current_gesture = 'Pointing Right' if delta_x > 0 else 'Pointing Left'
                            else: 
                                current_gesture = 'Pointing Down' if delta_y > 0 else 'Pointing Up'
                            gesture_text = f'Gesture: {current_gesture}'
                        else:
                            if index < len(latest_result.gestures) and latest_result.gestures[index]:
                                raw_name = latest_result.gestures[index][0].category_name
                                gesture_text = f"Gesture: {raw_name.replace('_', ' ')}"
                                current_gesture = raw_name
                            else: 
                                gesture_text = 'Gesture: None'
                                current_gesture = 'None' 

                        # Prevent background noise/multi-hands from blanking out true intent values
                        if current_gesture != 'None' or gesture_name == 'None':
                            gesture_name = current_gesture

                        # Render Head-Up-Display metadata overlays
                        wx, wy = int(wrist.x * w), int(wrist.y * h)
                        cv2.putText(frame, gesture_text, (wx - 50, wy - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, f"Count: {fingers} fingers", (wx - 50, wy - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

                # Autopilot automated control routines execution pipeline
                if not keyboard_active:
                    automated_rc = [0, 0, 0, 0] 

                    if mode == 'gesture':
                        match gesture_name:
                            case 'Closed_Fist':
                                automated_rc = [0, 0, 0, 0]
                            case 'Pointing Down':
                                automated_rc[2] = -25
                            case 'Pointing Up':
                                automated_rc[2] = 25
                            case 'Pointing Left':
                                automated_rc[0] = -25
                            case 'Pointing Right':
                                automated_rc[0] = 25
                            case 'Thumbs_Up':
                                mode = 'tag'
                                print("MODE TRANSITION: [tag mode]")
                    
                    elif mode == 'tag' and len(detected_ids) > 0:
                        primary = detected_ids[0]

                        match primary:
                            case 0:
                                # Distance measurements based on flipped coordinates format adjustments
                                marker_corners = flipped_corners[0][0]
                                pixel_width = math.sqrt((marker_corners[0][0] - marker_corners[1][0])**2 + (marker_corners[0][1] - marker_corners[1][1])**2)

                                STOP_THRESHOLD_PIXELS = 130
                                if pixel_width < STOP_THRESHOLD_PIXELS: automated_rc[1] = 20
                                else: automated_rc[1] = 0
                            case 1: automated_rc[1] = -20
                            case 2: automated_rc[3] = -20
                            case 3: automated_rc[3] = 20
                            case 4: 
                                automated_rc[3] = 0
                                mode = 'gesture'
                                print("MODE TRANSITION: [gesture mode]")
                    
                    rc_control = automated_rc

                # Monitor simulated Tello command dispatch changes
                if rc_control != last_rc_control:
                    print(f"TELLO TX DISPATCH Vector Array: {rc_control}")
                    last_rc_control = rc_control.copy()

                # Transform opencv workspace matrices into active Pygame display surfaces
                frame_render = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_render = frame_render.swapaxes(0, 1)

                frame_surface = pygame.surfarray.make_surface(frame_render)
                screen.blit(frame_surface, (0, 0))
                pygame.display.flip()

            clock.tick(30)
    except Exception as e: 
        print(f"Simulator Error: {e}")
    finally:
        cap.release()
        pygame.quit()
        print("Simulator Session Ended Safely.")