import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Global container for the asynchronous worker thread results
latest_result = None

def save_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# Configurations for Tasks API
model_path = 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,  # Captures up to 2 hands tracking simultaneously
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    result_callback=save_result
)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Video device could not be initialized.")
    exit()

with vision.HandLandmarker.create_from_options(options) as landmarker:
    print("Processing pipeline active. Press 'q' to terminate execution.")
    
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
        landmarker.detect_async(mp_image, timestamp_ms)

        if latest_result and latest_result.hand_landmarks:
            for index, hand_landmarks in enumerate(latest_result.hand_landmarks):
                
                # Render tracking dots on the image frame
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                wrist = hand_landmarks[0]
                fingers_open = []

                # 1. Thumb tracking logic (uses distance relative to pinky finger base knuckle to detect extension)
                thumb_tip = hand_landmarks[4]
                pinky_base = hand_landmarks[17]
                thumb_dist = math.sqrt((thumb_tip.x - pinky_base.x)**2 + (thumb_tip.y - pinky_base.y)**2)
                
                if thumb_dist > 0.3:  # Static spatial ratio threshold
                    fingers_open.append(1)
                else:
                    fingers_open.append(0)

                # 2. Structural tracking loop for standard fingers (Index, Middle, Ring, Pinky)
                # Compares absolute Tip-to-Wrist distance against Knuckle-to-Wrist distance
                finger_tip_ids = [8, 12, 16, 20]
                for tip_id in finger_tip_ids:
                    tip = hand_landmarks[tip_id]
                    knuckle = hand_landmarks[tip_id - 2] # Joint directly below tip
                    
                    dist_tip_to_wrist = math.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
                    dist_knuckle_to_wrist = math.sqrt((knuckle.x - wrist.x)**2 + (knuckle.y - wrist.y)**2)
                    
                    if dist_tip_to_wrist > dist_knuckle_to_wrist:
                        fingers_open.append(1) # Finger is extended
                    else:
                        fingers_open.append(0) # Finger is curled into palm

                total_fingers = sum(fingers_open)
                
                # Assign string mappings based on state arrays
                if total_fingers == 5:
                    gesture = "Open Hand"
                elif total_fingers == 0:
                    gesture = "Fist"
                elif fingers_open == [0, 1, 0, 0, 0]:
                    gesture = "Pointing"
                elif fingers_open == [0, 1, 1, 0, 0]:
                    gesture = "Victory"
                else:
                    gesture = f"Fingers: {total_fingers}"

                # Render Text HUD next to wrist anchor point
                wx, wy = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(frame, f"H{index}: {gesture}", (wx - 50, wy - 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Robust Multi-Hand Signal Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()