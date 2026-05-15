from djitellopy import Tello
import time

tello = Tello()
tello.connect()

# 1. Display a static pattern (an 'X' shape)
# Each character represents one of the 64 pixels (row by row)
pattern = "r000000rb00000br00bb000000bb000000bb0000rb0000brr000000r"
tello.send_control_command(f"mled g {pattern}")

time.sleep(3)

# 2. Display a scrolling string
# 'l' = left, 'r' = right, 'u' = up, 'd' = down
# Syntax: mled s [direction] [color] [speed] [text]
tello.send_control_command("mled s l r 2.5 HELLO")

time.sleep(5)

# 3. Clear the display
tello.send_control_command("mled off")

tello.end()