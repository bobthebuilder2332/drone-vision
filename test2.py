import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Global container for the asynchronous worker thread results
latest_result = None

def save_result(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# --- CAMERA INITIALIZATION ---
cap = cv2.VideoCapture(0)

# --- ARUCO MARKER DETECTION INITIALIZATION ---
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# --- MEDIAPIPE GESTURE RECOGNIZER INITIALIZATION ---
model_path = 'gesture_recognizer.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    result_callback=save_result
)

frame_counter = 0
current_mode = 'tag'  # Initial mode: 'tag' or 'gesture'

print("=" * 60)
print("Press 't' to TOGGLE between TAG and GESTURE modes.")
print("Press 'q' to QUIT.")
print("=" * 60)

with vision.GestureRecognizer.create_from_options(options) as recognizer:
    while True: 
        frame_counter += 1
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        gesture_name = "N/A (Tag Mode)"
        total_fingers = 0
        detected_ids = []

        # ==========================================
        # 1. ARUCO TAG MODE
        # ==========================================
        if current_mode == 'tag':
            corners, ids, _ = detector.detectMarkers(frame)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                detected_ids = ids.flatten()
            
            # Draw Mode HUD on frame
            cv2.putText(frame, "MODE: TAG DETECTION", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # ==========================================
        # 2. MEDIAPIPE HAND GESTURE MODE
        # ==========================================
        elif current_mode == 'gesture':
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            timestamp_ms = frame_counter * 33
            recognizer.recognize_async(mp_image, timestamp_ms)

            # Draw Mode HUD on frame
            cv2.putText(frame, "MODE: GESTURE DETECTION", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            if latest_result and latest_result.hand_landmarks:
                gesture_name = "None"
                for index, hand_landmarks in enumerate(latest_result.hand_landmarks):
                    
                    # Render tracking joints
                    for lm in hand_landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                    wrist = hand_landmarks[0]
                    index_base = hand_landmarks[5]
                    pinky_base = hand_landmarks[17]
                    
                    hand_scale = math.sqrt((index_base.x - pinky_base.x)**2 + (index_base.y - pinky_base.y)**2)

                    # --- ADVANCED THUMB DETECTION ---
                    thumb_tip = hand_landmarks[4]
                    thumb_ip = hand_landmarks[3]

                    tip_to_index_base = math.sqrt((thumb_tip.x - index_base.x)**2 + (thumb_tip.y - index_base.y)**2)
                    ip_to_index_base = math.sqrt((thumb_ip.x - index_base.x)**2 + (thumb_ip.y - index_base.y)**2)

                    if (tip_to_index_base / hand_scale) < 0.42 or (ip_to_index_base / hand_scale) < 0.38:
                        thumb_open = 0
                    else:
                        thumb_open = 1

                    # --- OMNIDIRECTIONAL FINGER DETECTION ---
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

                    total_fingers = sum(fingers_open)

                    # --- DIRECTIONAL POINTING CLASSIFICATION ---
                    if fingers_open == [0, 1, 0, 0, 0] or fingers_open == [1, 1, 0, 0, 0]:
                        idx_tip = hand_landmarks[8]
                        idx_base = hand_landmarks[5]
                        
                        delta_x = idx_tip.x - idx_base.x
                        delta_y = idx_tip.y - idx_base.y

                        if abs(delta_x) > abs(delta_y):
                            gesture_name = "Pointing Right" if delta_x > 0 else "Pointing Left"
                        else:
                            gesture_name = "Pointing Down" if delta_y > 0 else "Pointing Up"
                        
                        gesture_text = f"Gesture: {gesture_name}"
                    else:
                        if index < len(latest_result.gestures) and latest_result.gestures[index]:
                            raw_name = latest_result.gestures[index][0].category_name
                            gesture_text = f"Gesture: {raw_name.replace('_', ' ')}"
                            gesture_name = raw_name 
                        else:
                            gesture_text = "Gesture: Calculating"

                    # --- HUD RENDERING ON FRAME ---
                    wx, wy = int(wrist.x * w), int(wrist.y * h)
                    cv2.putText(frame, gesture_text, (wx - 50, wy - 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Count: {total_fingers} fingers", (wx - 50, wy - 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # Print current system status
        if current_mode == 'tag':
            print(f"[TAG MODE] Detected Tags: {detected_ids}")
        else:
            print(f"[GESTURE MODE] Gesture: {gesture_name} | Fingers: {total_fingers}")

        # Display output feed locally
        window_title = f"Vision Tracking | Active Mode: {current_mode.upper()}"
        cv2.imshow(window_title, frame)
        
        # Handle Key Inputs
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):  # Toggle mode key
            cv2.destroyAllWindows()  # Clears window title bar data cleanly on swap
            current_mode = 'gesture' if current_mode == 'tag' else 'tag'
            print(f"\n>>> Swapped mode to: {current_mode.upper()} <<<\n")

cap.release()
cv2.destroyAllWindows()