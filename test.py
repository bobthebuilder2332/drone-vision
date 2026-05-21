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

# Configurations for Gesture Recognizer Tasks API
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

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Video device could not be initialized.")
    exit()

with vision.GestureRecognizer.create_from_options(options) as recognizer:
    print("Pipeline Active. Press 'q' to terminate execution.")
    
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Prepare frame format for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        frame_count += 1
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC)) or (frame_count * 33)
        recognizer.recognize_async(mp_image, timestamp_ms)

        if latest_result and latest_result.hand_landmarks:
            for index, hand_landmarks in enumerate(latest_result.hand_landmarks):
                
                # Render tracking joints
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                wrist = hand_landmarks[0]
                index_base = hand_landmarks[5]
                pinky_base = hand_landmarks[17]
                
                # Establish fundamental scale of the hand in 2D space
                hand_scale = math.sqrt((index_base.x - pinky_base.x)**2 + (index_base.y - pinky_base.y)**2)

                # --- ADVANCED THUMB DETECTION ---
                thumb_tip = hand_landmarks[4]
                thumb_ip = hand_landmarks[3]
                thumb_mcp = hand_landmarks[2]

                # Distance from thumb tip to index base knuckle
                tip_to_index_base = math.sqrt((thumb_tip.x - index_base.x)**2 + (thumb_tip.y - index_base.y)**2)
                # Distance from thumb IP joint to index base knuckle
                ip_to_index_base = math.sqrt((thumb_ip.x - index_base.x)**2 + (thumb_ip.y - index_base.y)**2)

                # When flat against the palm, the thumb tip and IP joint collapse inward toward index base
                if (tip_to_index_base / hand_scale) < 0.42 or (ip_to_index_base / hand_scale) < 0.38:
                    thumb_open = 0
                else:
                    thumb_open = 1

                # --- OMNIDIRECTIONAL FINGER DETECTION ---
                # Measure distance relative to wrist to evaluate extension state regardless of hand rotation
                finger_tips = [8, 12, 16, 20]
                finger_pips = [6, 10, 14, 18] # PIP joints (second joint from base)
                
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
                # Check state array: Thumb down/flat (0), Index UP (1), others DOWN (0,0,0)
                if fingers_open == [0, 1, 0, 0, 0] or fingers_open == [1, 1, 0, 0, 0]:
                    idx_tip = hand_landmarks[8]
                    idx_base = hand_landmarks[5]
                    
                    delta_x = idx_tip.x - idx_base.x
                    delta_y = idx_tip.y - idx_base.y

                    # Determine primary vector component
                    if abs(delta_x) > abs(delta_y):
                        gesture_name = "Pointing Right" if delta_x > 0 else "Pointing Left"
                    else:
                        gesture_name = "Pointing Down" if delta_y > 0 else "Pointing Up"
                    
                    gesture_text = f"Gesture: {gesture_name}"
                else:
                    # Fallback to standard ML gesture recognizer strings if not custom pointing
                    if index < len(latest_result.gestures) and latest_result.gestures[index]:
                        raw_name = latest_result.gestures[index][0].category_name
                        gesture_text = f"Gesture: {raw_name.replace('_', ' ')}"
                    else:
                        gesture_text = "Gesture: Calculating"

                # --- HUD RENDERING ---
                wx, wy = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(frame, gesture_text, (wx - 50, wy - 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Count: {total_fingers} fingers", (wx - 50, wy - 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Omnidirectional Tracking Pipeline', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()