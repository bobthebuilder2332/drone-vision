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

                # 1. FIXED THUMB LOGIC: Independent of Handedness, Mirroring, or Palm Orientation
                # Uses the horizontal distance between Thumb Tip (4) and Index Finger Base (5)
                # normalized by the distance between Index Base (5) and Pinky Base (17) to scale for camera distance.
                thumb_tip = hand_landmarks[4]
                index_base = hand_landmarks[5]
                pinky_base = hand_landmarks[17]

                # Hand scale metric (distance between knuckles across the palm)
                hand_scale = math.sqrt((index_base.x - pinky_base.x)**2 + (index_base.y - pinky_base.y)**2)
                
                # Horizontal distance from thumb tip to index base
                thumb_to_index_x = abs(thumb_tip.x - index_base.x)

                # If the horizontal distance is greater than 38% of the hand's width, the thumb is open
                if (thumb_to_index_x / hand_scale) > 0.38:
                    fingers_open.append(1)
                else:
                    fingers_open.append(0)

                # 2. Corrected Finger tracking loop (Index, Middle, Ring, Pinky)
                # Compares Y-coordinates of Tip vs PIP joint (Joint 2 steps below tip)
                finger_tip_ids = [8, 12, 16, 20]
                for tip_id in finger_tip_ids:
                    tip = hand_landmarks[tip_id]
                    pip = hand_landmarks[tip_id - 2]
                    
                    # MediaPipe Y-axis increases downward. Lower Y value means higher in frame.
                    if tip.y < pip.y:
                        fingers_open.append(1)
                    else:
                        fingers_open.append(0)

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