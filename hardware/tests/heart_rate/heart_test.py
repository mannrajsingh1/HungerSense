
import time
import statistics
from grove.adc import ADC

adc = ADC(address=0x04)
CHANNEL = 0

print("Heart sensor test")
print("Keep your finger still on the sensor")
print("Press Ctrl+C to stop")

try:
    while True:
        samples = []

        for _ in range(500):
            value = adc.read_raw(CHANNEL)
            samples.append(value)
            time.sleep(0.01)

        minimum = min(samples)
        maximum = max(samples)
        average = statistics.mean(samples)

        print(
            f"Min: {minimum} | "
            f"Max: {maximum} | "
            f"Average: {average:.1f} | "
            f"Range: {maximum - minimum}"
        )

except KeyboardInterrupt:
    print("\nTest stopped")
