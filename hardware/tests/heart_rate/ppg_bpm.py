
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks, medfilt
from grove.adc import ADC

adc = ADC(address=0x04)

DURATION = 25
FS = 50
INVERT = False  # Set True if your PPG peaks are inverted

input("Place finger gently on sensor, then press Enter...")

times, readings = [], []
start = time.monotonic()

print("Collecting optical PPG signal...")

while time.monotonic() - start < DURATION:
    times.append(time.monotonic() - start)
    readings.append(adc.read_raw(0))
    time.sleep(0.01)

times = np.array(times)
readings = np.array(readings)
dropouts = np.sum(readings < 100)

print(f"Total samples: {len(readings)}")
print(f"Dropouts: {dropouts}")
print(f"Dropout percentage: {100 * dropouts / len(readings):.2f}%")
print(f"Raw min: {np.min(readings)}")
print(f"Raw max: {np.max(readings)}")

# Resample to a regular sampling rate
t = np.arange(0, times[-1], 1 / FS)
raw = np.interp(t, times, readings)

# Remove isolated spikes
clean = medfilt(raw, kernel_size=5)

# Band-pass filter: typical pulse frequencies
sos = butter(
    3, [0.7, 3.0],
    btype="bandpass",
    fs=FS,
    output="sos"
)

filtered = sosfiltfilt(sos, clean)

if INVERT:
    filtered = -filtered

# Detect candidate peaks
peaks, _ = find_peaks(
    filtered,
    distance=int(0.4 * FS),
    prominence=max(2, np.std(filtered) * 0.8)
)

# Exclude filter edge artifacts
peaks = peaks[
    (peaks > FS) &
    (peaks < len(t) - FS)
]

# Plot filtered waveform and detected peaks
plt.figure(figsize=(12, 4))
plt.plot(t, filtered, label="Filtered PPG")
plt.plot(t[peaks], filtered[peaks], "ro",
         label="Detected peaks")
plt.xlabel("Time (seconds)")
plt.ylabel("Filtered signal (ADC units)")
plt.title("PPG Heartbeat Detection")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ppg_waveform.png")
plt.close()

print("Saved ppg_waveform.png")

# Estimate BPM only if signal quality looks reasonable
if np.any(readings < 100):
    print("Signal dropouts detected; BPM unreliable.")
elif np.std(filtered) < 2 or len(peaks) < 10:
    print("Insufficient pulse signal for BPM.")
else:
    intervals = np.diff(t[peaks])
    bpm = 60 / np.median(intervals)

    cv = np.std(intervals) / np.mean(intervals)

    if (
        np.any(intervals < 0.4) or
        np.any(intervals > 1.5) or
        cv > 0.20
    ):
        print("Irregular peak detection; BPM unreliable.")
    else:
        print(f"Estimated BPM: {bpm:.1f}")
        print(f"Detected peaks: {len(peaks)}")
        print("Experimental estimate only.")
