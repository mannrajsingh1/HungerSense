
import time
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
from grove.adc import ADC

adc = ADC(address=0x04)
CHANNEL = 0
DURATION = 25  # seconds
FS = 50        # resampling rate

input("Place finger on sensor, then press Enter...")

times = []
values = []

print("Recording heartbeat signal...")

start = time.monotonic()

while time.monotonic() - start < DURATION:
    t = time.monotonic() - start
    value = adc.read_raw(CHANNEL)

    times.append(t)
    values.append(value)

    time.sleep(0.01)

times = np.array(times)
values = np.array(values)

print(f"Collected {len(values)} samples")

# Check for obvious signal problems
if np.std(values) < 10:
    print("Signal nearly constant. No reliable BPM.")
    raise SystemExit

if np.mean(values < 100) > 0.01:
    print("Too many near-zero readings. Check wiring.")
    raise SystemExit

# Resample onto a uniform time grid
uniform_times = np.arange(0, times[-1], 1 / FS)
signal = np.interp(uniform_times, times, values)

# Band-pass filter: 0.7-3 Hz
sos = butter(
    3, [0.7, 3.0],
    btype="bandpass",
    fs=FS,
    output="sos"
)
filtered = sosfiltfilt(sos, signal)

# Detect candidate heartbeat peaks
peaks, _ = find_peaks(
    filtered,
    distance=int(FS * 0.45),
    prominence=max(np.std(filtered) * 0.9, 2)
)

# Ignore filter artifacts near the boundaries
peaks = peaks[
    (peaks > FS) &
    (peaks < len(filtered) - FS)
]

if len(peaks) < 8:
    print("Not enough pulse peaks for reliable BPM.")
else:
    beat_times = uniform_times[peaks]
    intervals = np.diff(beat_times)

    bpm = 60 / np.median(intervals)
    variability = np.std(intervals) / np.mean(intervals)

    if (
        np.any(intervals < 0.4) or
        np.any(intervals > 1.5) or
        variability > 0.20
    ):
        print("Unstable pulse intervals.")
        print("BPM unavailable: check sensor placement.")
    else:
        print(f"Estimated BPM: {bpm:.1f}")
        print(f"Detected peaks: {len(peaks)}")
        print("Experimental estimate; verify independently.")
