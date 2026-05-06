from djitellopy import Tello # djitellopy2 uses the same module name
import cv2
import pygame


# Initialize the Tello drone
drone = Tello()
drone.connect()
print(f"Battery: {drone.get_battery()}%")
drone.streamon()
frame_read = drone.get_frame_read() # Start the video stream and get the frame reader object (runs in background)


# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Tello Drone Feed")
clock = pygame.time.Clock()


# Drone control variables
speed = 50
moveForward, strafe, ascend, rotate = 0, 0, 0, 0


# Main loop
running = True
while running:
      # Input handling
      for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False 
            if event.type == pygame.KEYDOWN:
                  if event.key == pygame.K_t: drone.takeoff()
                  if event.key == pygame.K_l: drone.land()
                  if event.key == pygame.K_e: drone.emergency() # Emergency stop and cuts all motors
                  if event.key == pygame.K_q: running = False
                  
                  if event.key == pygame.K_w: moveForward = speed
                  if event.key == pygame.K_s: moveForward = -speed
                  if event.key == pygame.K_a: strafe = -speed
                  if event.key == pygame.K_d: strafe = speed
                  if event.key == pygame.K_LEFT: rotate = -speed
                  if event.key == pygame.K_RIGHT: rotate = speed
                  if event.key == pygame.K_UP: ascend = speed
                  if event.key == pygame.K_DOWN: ascend = -speed
            if event.type == pygame.KEYUP:
                  if event.key in [pygame.K_w, pygame.K_s]: moveForward = 0
                  if event.key in [pygame.K_a, pygame.K_d]: strafe = 0
                  if event.key in [pygame.K_LEFT, pygame.K_RIGHT]: rotate = 0
                  if event.key in [pygame.K_UP, pygame.K_DOWN]: ascend = 0

      drone.send_rc_control(strafe, moveForward, ascend, rotate) # Send the commands to the drone

      # Get the video frame from the drone
      frame = frame_read.frame # Get the latest frame from the video stream
      if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color from BGR (OpenCV) to RGB (Pygame)
            frame = frame.swapaxes(0, 1)  # Swap axes from height x width (OpenCV) to width x height (Pygame)
   
            # Convert the frame to a Pygame surface and display it
            frame_surface = pygame.surfarray.make_surface(frame)
            screen.blit(frame_surface, (0, 0))
            pygame.display.flip() # Same thing as .update() for entire window

      clock.tick(30) # Limit to 30 FPS; don't overburden cpu

# Clean up
drone.land()
drone.streamoff()
pygame.quit()