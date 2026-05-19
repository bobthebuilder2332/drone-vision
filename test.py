import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Global container for the asynchronous worker thread results
latest_result = None

def save_result(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# Configurations for Gesture Recognizer Tasks API
# REQUIRES: gesture_recognizer.task model file in the same directory
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
    print("Gesture pipeline active. Press 'q' to terminate execution.")
    
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

        if latest_result and latest_result.gestures:
            # Loop through detected hands
            for index, hand_landmarks in enumerate(latest_result.hand_landmarks):
                
                # Draw tracking joints
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                # Get the gesture name and confidence score
                if index < len(latest_result.gestures) and latest_result.gestures[index]:
                    top_gesture = latest_result.gestures[index][0]
                    gesture_name = top_gesture.category_name
                    confidence = top_gesture.score
                    
                    # MediaPipe labels lack of explicit gesture as "None"
                    if gesture_name == "None":
                        gesture_text = "Unknown Gesture"
                    else:
                        gesture_text = f"{gesture_name} ({confidence*100:.1f}%)"
                else:
                    gesture_text = "Tracking..."

                # Render output text at the wrist point (Landmark 0)
                wrist = hand_landmarks[0]
                wx, wy = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(frame, f"H{index}: {gesture_text}", (wx - 50, wy - 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Built-in MediaPipe Gesture Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()