from djitellopy import Tello # djitellopy2 uses the same module name ('djitellopy')
import cv2
import cv2.aruco as aruco # Module for ArUco marker detection
import pygame
import threading # Used for takeoff  and landing to prevent the program from hanging (those are blocking calls)


# Initialize the Tello drone
drone = Tello()
drone.connect()

battery = f"Battery: {drone.get_battery()}%"
print(f"{battery:=^40}")

drone.streamon()
frame_read = drone.get_frame_read() # Start the video stream and get the frame reader object (runs in background)


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
try:
      running = True
      while running: 
            # Input handling
            for event in pygame.event.get():
                  if event.type == pygame.QUIT: running = False 
                  if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_t: threading.Thread(target=drone.takeoff, daemon=True).start()
                        if event.key == pygame.K_l: threading.Thread(target=drone.land, daemon=True).start()
                        if event.key == pygame.K_e: drone.emergency() # Emergency stop and cuts all motors
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

            # Only send command if there is a change
            if rc_control != last_rc_control:
                  drone.send_rc_control(*rc_control) # Send the commands to the drone; * is unpack operator to pass through values instead of list
                  last_rc_control = rc_control.copy() # Use .copy() to copy instead of reference

            # Get the video frame from the drone
            frame = frame_read.frame # Get the latest frame from the video stream
            if frame is not None:
                  battery = f"Battery: {drone.get_battery()}%"
                  pygame.display.set_caption(f"Tello Drone Feed | {battery}") # Show battery percentage in window name
                  
                  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color from BGR (OpenCV) to RGB (Pygame)
                  frame = frame.swapaxes(0, 1)  # Swap axes from height x width (OpenCV) to width x height (Pygame)
      
                  # Convert the frame to a Pygame surface and display it
                  frame_surface = pygame.surfarray.make_surface(frame)
                  screen.blit(frame_surface, (0, 0))
                  pygame.display.flip() # Same thing as .update() for entire window

            clock.tick(30) # Limit to 30 FPS; don't overburden cpu
except Exception as e:
      print(f"Error: {e}")
finally:
      # Clean up
      drone.land()
      drone.streamoff()
      pygame.quit()
      drone.end()