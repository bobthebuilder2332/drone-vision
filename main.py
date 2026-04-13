import pygame
import cv2
import numpy as np
from djitellopy import Tello

# Initialization and setup
WIDTH, HEIGHT = 960, 720
pygame.init() # Stats pygame engine
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Drone Video Feed")
clock = pygame.time.Clock() # Use clock to control frame rate and reduce unneccessary CPU usage
tello = Tello()

# Checks for connection to drone over WiFi
try:
    tello.connect() # Tries to connect to drone
    tello.streamon() # Starts the video camera
    print(f"Connected! Battery: {tello.get_battery()}%")
except Exception as e: # If connection fails
    print(f"Could not connect to drone. Error: {e}")
    pygame.quit() # Quits pygame to free system resources
    exit() # Exits the program

# Get the video stream background worker; needs this to get latest frame without lag
frame_read = tello.get_frame_read()

# --- 3. MAIN LOOP ---
running = True
while running:
    for event in pygame.event.get(): # Event handling
        if event.type == pygame.QUIT: # Close / quit window
            running = False
        
        if event.type == pygame.KEYDOWN:
            # Emergency landing (spacebar)
            if event.key == pygame.K_SPACE:
                print("EMERGENCY LANDING...")
                tello.land()
            # Takeoff (t key)
            if event.key == pygame.K_t:
                print("TAKEOFF!")
                tello.takeoff()

    # Vision processing
    img = frame_read.frame # Gets the current frame from the drone
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert color space from BGR to RGB
    
    img = cv2.resize(img, (WIDTH, HEIGHT)) # Resize to fit window
    
    # Swap x and y because opencv does h,w and pygame does w,h
    # Also flip image because tarnspose causes image to be mirrored
    img = np.rot90(img)
    img = pygame.surfarray.make_surface(img) # Convert numpy array to pygame surface
    img = pygame.transform.flip(img, True, False)


    screen.blit(img, (0, 0)) # Copy the frame onto the screen
    pygame.display.update()

    clock.tick(30) # Set to 30 FPS to match drone and reduce CPU load

# Remeber to clean up and free system resources
tello.streamoff()
tello.land() # Prevents the drone from hovering if program is closed while flying
pygame.quit()