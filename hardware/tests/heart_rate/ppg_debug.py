
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from grove.adc import ADC
from scipy.signal import butter, sosfiltfilt

adc = ADC(address=0x04)

print("Place finger on sensor and hold still.")
input("Press Enter to start...")

times = []
values = []
start = time.monotonic()

while time.monotonic() - start < 20:
    times.append(time.monotonic() - start)
    values.append(adc.read_raw(0))
    time.sleep(0.02)

t = np.array(times)
raw = np.array(values, dtype=float)

# Uniform sampling
fs = 50
grid = np.arange(t[0], t[-1], 1 / fs)
signal = np.interp(grid, t, raw)

# Band-pass filter
sos = butter(
    3, [0.7, 3.0],
    btype="bandpass",
    fs=fs,
    output="sos"
)

filtered = sosfiltfilt(sos, signal)

from scipy.signal import find_peaks

# Analyze the relatively stable section
mask = (grid >= 11) & (grid <= 18)

t_clean = grid[mask]
y_clean = filtered[mask]

# Detect candidate heartbeat peaks
peaks, properties = find_peaks(
    y_clean,
    distance=int(0.45 * fs),
    prominence=max(5, np.std(y_clean) * 0.6)
)

peak_times = t_clean[peaks]

if len(peak_times) >= 6:
    intervals = np.diff(peak_times)

    print("\n--- Peak Detection Analysis ---")

    print("Peak times:", np.round(peak_times, 2))

    print("Intervals:", np.round(intervals, 2))

    print("Median interval:", round(np.median(intervals), 3))

    bpm = 60 / np.median(intervals)
    print(f"Candidate BPM: {bpm:.1f}")

    cv = np.std(intervals) / np.mean(intervals)
    print(f"Interval variability: {cv:.2f}")

    if cv < 0.15:
        print("Regular candidate peaks; verify with reference.")
    else:
        print("Unstable peaks: BPM not reliable.")

    bpm = 60 / np.median(intervals)
    variability = np.std(intervals) / np.mean(intervals)

    print(f"Candidate BPM: {bpm:.1f}")
    print(f"Detected peaks: {len(peaks)}")
    print(f"Interval variability: {variability:.2f}")

    if variability > 0.20:
        print("WARNING: Inconsistent peak intervals")
    else:
        print("Intervals are fairly regular; verify BPM.")
else:
    print("Insufficient peaks for BPM estimation")

# Plot the selected section
plt.figure(figsize=(12, 4))
plt.plot(t_clean, y_clean, label="Filtered PPG")
plt.scatter(
    peak_times,
    y_clean[peaks],
    color="red",
    label="Candidate heartbeats"
)
plt.xlabel("Time (seconds)")
plt.ylabel("Filtered ADC")
plt.title("PPG Peak Detection: 11-18 seconds")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ppg_peaks.png", dpi=150)
plt.close()

print("Saved ppg_peaks.png")

fig, axes = plt.subplots(2, 1, figsize=(12, 7))

axes[0].plot(grid, signal)
axes[0].set_title("RAW PPG Signal")
axes[0].set_ylabel("ADC value")
axes[0].grid(True)

axes[1].plot(grid, filtered)
axes[1].set_title("Filtered PPG Signal")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Filtered ADC")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("ppg_debug.png", dpi=150)

print("Saved ppg_debug.png")
print("Raw min:", np.min(raw))
print("Raw max:", np.max(raw))
print("Raw std:", np.std(raw))
print("Near-zero samples:", np.sum(raw < 100))
