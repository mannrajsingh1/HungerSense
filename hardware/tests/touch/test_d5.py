from gpiozero import DigitalInputDevice
from time import sleep

sensor = DigitalInputDevice(5)

print("Reading Grove D5 / GPIO5. Ctrl+C to stop.")

try:
    while True:
        print("D5:", sensor.value)
        sleep(0.1)
except KeyboardInterrupt:
    print("\nStopped.")
