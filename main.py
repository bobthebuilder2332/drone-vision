from djitellopy import Tello

# Replace with your computer's actual IP on the Tello Wi-Fi network
# You can find this in your Network Settings
drone = Tello(host='192.168.10.1') 

try:
    drone.connect()
    print(f"Battery: {drone.get_battery()}%")
except Exception as e:
    print(f"Connection Failed: {e}")