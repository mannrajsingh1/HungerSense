
from grove.adc import ADC
import time
import statistics

adc = ADC(address=0x04)

for condition in ["NO FINGER", "FINGER ON SENSOR"]:
    input(f"\n{condition}: Press Enter to start...")

    values = []

    for i in range(500):
        values.append(adc.read_raw(0))
        time.sleep(0.01)

    print(f"\n{condition}")
    print("Average:", round(statistics.mean(values), 1))
    print("Median:", statistics.median(values))
    print("Minimum:", min(values))
    print("Maximum:", max(values))
    print("Std Dev:", round(statistics.stdev(values), 1))
    print("Near-zero readings:", sum(v < 100 for v in values))

print("\nTest completed!")
