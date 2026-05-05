from djitellopy import Tello # djitellopy2 uses the same module name
import cv2
import pygame


# Initialize the Tello drone
drone = Tello()
drone.connect()
print(f"Battery: {drone.get_battery()}%")

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Tello Drone Feed")

# Start video stream
drone.streamon()

running = True
while running:
      # Input handling
      for event in pygame.event.get():
         if event.type == pygame.QUIT: running = False 
   
      # Get the video frame from the drone
      frame = drone.get_frame_read().frame
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color for Pygame
   
      # Convert the frame to a Pygame surface and display it
      frame_surface = pygame.surfarray.make_surface(frame)
      screen.blit(frame_surface, (0, 0))
      pygame.display.flip() # Same thing as .update() for entire window

# Clean up
drone.streamoff()
pygame.quit()