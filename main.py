from djitellopy import Tello # djitellopy2 uses the same module name ('djitellopy')
import cv2
import pygame
import threading # Used for takeoff and landing to prevent the program from hanging (those are blocking calls)
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Global container for the asynchronous worker thread results
latest_result = None

def save_result(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# Initialize the Tello drone
drone = Tello()
drone.connect()

battery = f"Battery: {drone.get_battery()}%"
print(f"{battery:=^40}")

drone.streamon()
frame_read = drone.get_frame_read() # Start the video stream and get the frame reader object (runs in background)


# Initialize the ArUco marker detection
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50) # Get the predefined dictionary (use 50 because tradeoff variety for speed)
aruco_params = cv2.aruco.DetectorParameters() # Default parameters
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params) # Create the ArUco marker detector using the specified dictionary and parameters
MARKER_SIZE = .1 # Physical size of marker is 100 mm

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

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((960, 720))
pygame.display.set_caption(f"Tello Drone Feed | {battery}")
clock = pygame.time.Clock()


# Drone control variables
speed = 50
rc_control = [0, 0, 0, 0] # strafe, moveForward, ascend, rotate
last_rc_control = [0, 0, 0, 0]


# Main loop
with vision.GestureRecognizer.create_from_options(options) as recognizer:
    try:
          running = True
          frame_counter = 0
          while running: 
                frame_counter += 1

                # Reset keyboard tracking state at the beginning of each frame loop
                keyboard_active = False

                # Input handling
                for event in pygame.event.get():
                      if event.type == pygame.QUIT: running = False 
                      if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_t: threading.Thread(target=drone.takeoff, daemon=True).start()
                            if event.key == pygame.K_l: threading.Thread(target=drone.land, daemon=True).start()
                            if event.key == pygame.K_e: threading.Thread(target=drone.emergency, daemon=True).start() # Emergency stop and cuts all motors
                            if event.key == pygame.K_q: running = False
                            
                            if event.key == pygame.K_w: rc_control[1] = speed
                            if event.key == pygame.K_s: rc_control[1] = -speed
                            if event.key == pygame.K_a: rc_control[0] = -speed
                            if event.key == pygame.K_d: rc_control[0] = speed
                            if event.key == pygame.K_LEFT: rc_control[3] = -speed
                            if event.key == pygame.K_RIGHT: rc_control[3] = speed
                            if event.key == pygame.K_UP: rc_control[2] = speed
                            if event.key == pygame.K_DOWN: rc_control[2] = -speed
                      if event.type == pygame.KEYUP:
                            if event.key in [pygame.K_w, pygame.K_s]: rc_control[1] = 0
                            if event.key in [pygame.K_a, pygame.K_d]: rc_control[0] = 0
                            if event.key in [pygame.K_LEFT, pygame.K_RIGHT]: rc_control[3] = 0
                            if event.key in [pygame.K_UP, pygame.K_DOWN]: rc_control[2] = 0

                # Check if manual overrides are actively being driven by keyboard inputs
                if rc_control != [0, 0, 0, 0]:
                      keyboard_active = True

                # Get the video frame from the drone
                frame = frame_read.frame # Get the latest frame from the video stream
                if frame is not None:
                      h, w, _ = frame.shape
                      gesture_name = "None" # Default fallback state for safety tracking
                      total_fingers = 0
                      
                      # Only run battery check every 5 seconds
                      if frame_counter % 150 == 0:
                            battery = f"Battery: {drone.get_battery()}%"
                            pygame.display.set_caption(f"Tello Drone Feed | {battery}") # Show battery percentage in window name
                      
                      # Look for markers
                      corners, ids, _ = detector.detectMarkers(frame)
                      if ids is not None:
                            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                            detected_ids = ids.flatten()
                      else:
                            detected_ids = []

                      # --- MEDIAPIPE FRAME PROCESSING ENTRY ---
                      rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                      
                      timestamp_ms = frame_counter * 33
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

                              # --- HUD RENDERING ---
                              wx, wy = int(wrist.x * w), int(wrist.y * h)
                              cv2.putText(frame, gesture_text, (wx - 50, wy - 60), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                              cv2.putText(frame, f"Count: {total_fingers} fingers", (wx - 50, wy - 35), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

                      # ========================================================================
                      # FIXED AUTOMATED MOVEMENT LOGIC (GESTURES & ARUCO)
                      # ========================================================================
                      # Only processes automated commands if you aren't touching the keyboard
                      if not keyboard_active:
                          # Start fresh with a neutral hover loop vector baseline
                          automated_rc = [0, 0, 0, 0]
                          
                          # PRIORITY 1: Master Hand Gestures (Explicit Directions)
                          if gesture_name == "Closed_Fist":
                              automated_rc = [0, 0, 0, 0] 
                          elif gesture_name == "Pointing Up":
                              automated_rc[2] = 25  # Ascend
                          elif gesture_name == "Pointing Down":
                              automated_rc[2] = -25 # Descend
                          elif gesture_name == "Pointing Left":
                              automated_rc[0] = -25 # Strafe Left
                          elif gesture_name == "Pointing Right":
                              automated_rc[0] = 25  # Strafe Right
                              
                          # PRIORITY 2: ArUco Tag Autopilot (Runs if hand isn't overriding)
                          elif len(detected_ids) > 0:
                              primary_tag = detected_ids[0]
                              
                              if primary_tag == 0:
                                  marker_corners = corners[0][0] 
                                  pixel_width = math.sqrt((marker_corners[0][0] - marker_corners[1][0])**2 + 
                                                          (marker_corners[0][1] - marker_corners[1][1])**2)
                                  
                                  STOP_THRESHOLD_PIXELS = 130 
                                  if pixel_width < STOP_THRESHOLD_PIXELS:
                                      automated_rc[1] = 20  # Creep forward safely
                                  else:
                                      automated_rc[1] = 0   # Close proximity brake: Hover
                                      
                              elif primary_tag == 1:
                                  automated_rc[1] = -20 # Back away from obstacle
                              elif primary_tag == 2:
                                  automated_rc[3] = -30 # Scan left
                              elif primary_tag == 3:
                                  automated_rc[3] = 30  # Scan right
                              elif primary_tag == 4:
                                  threading.Thread(target=drone.land, daemon=True).start()
                          
                          # Assign calculated vectors back to primary tracking variable
                          rc_control = automated_rc
                      # ========================================================================
 
                # Only send command if there is a change
                if rc_control != last_rc_control:
                      drone.send_rc_control(*rc_control) # Send commands to drone
                      last_rc_control = rc_control.copy()

                if frame is not None:
                      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color from BGR (OpenCV) to RGB (Pygame)
                      frame = frame.swapaxes(0, 1)  # Swap axes from height x width (OpenCV) to width x height (Pygame)

                      # Convert the frame to a Pygame surface and display it
                      frame_surface = pygame.surfarray.make_surface(frame)
                      screen.blit(frame_surface, (0, 0))
                      pygame.display.flip()

                clock.tick(30) # Limit to 30 FPS
    except Exception as e:
          print(f"Error: {e}")
    finally:
          # Clean up
          drone.land()
          drone.streamoff()
          pygame.quit()
          drone.end()