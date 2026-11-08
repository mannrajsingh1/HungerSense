
import time
from collections import deque

import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks, medfilt
from grove.adc import ADC

# ====================================
# CONFIGURATION
# ====================================

ADC_ADDRESS = 0x04
CHANNEL = 0

WINDOW = 7.0
MIN_WINDOW = 4.0
UPDATE_INTERVAL = 1.0
SAMPLE_INTERVAL = 0.01
FS = 50

INVERT = False

adc = ADC(address=ADC_ADDRESS)
samples = deque()

sos = butter(
    3, [0.7, 3.0],
    btype="bandpass",
    fs=FS,
    output="sos"
)

# ====================================
# BPM DETECTION
# ====================================

def calculate_bpm():

    if len(samples) < 10:
        return None, "Collecting..."

    data = list(samples)

    times = np.array([s[0] for s in data])
    raw = np.array([s[1] for s in data], dtype=float)

    duration = times[-1] - times[0]

    if duration < MIN_WINDOW:
        return None, f"Collecting {duration:.1f}s..."

    # Identify missing or saturated readings
    valid = (raw >= 100) & (raw < 3900)

    if np.mean(~valid) > 0.01:
        return None, "Poor signal: check sensor contact"

    if np.sum(valid) < 20:
        return None, "Insufficient data"

    # Replace a few isolated bad samples
    clean = np.interp(times, times[valid], raw[valid])

    # Reject large discontinuities
    jumps = np.abs(np.diff(clean))
    if np.mean(jumps > 700) > 0.005:
        return None, "Signal disturbance"

    # Resample at 50Hz using real timestamps
    t = np.arange(times[0], times[-1], 1 / FS)

    signal = np.interp(t, times, clean)

    # Remove isolated spikes
    signal = medfilt(signal, kernel_size=5)

    # Filter optical pulse signal
    filtered = sosfiltfilt(sos, signal)

    if INVERT:
        filtered = -filtered

    amplitude = np.std(filtered)

    if amplitude < 3:
        return None, "Weak pulse signal"

    # Candidate heartbeat detection
    peaks, _ = find_peaks(
        filtered,
        distance=int(0.42 * FS),
        prominence=max(4, amplitude * 0.7)
    )

    # Ignore artifacts at filter boundaries
    peaks = peaks[
        (t[peaks] > t[0] + 0.3) &
        (t[peaks] < t[-1] - 0.3)
    ]

    if len(peaks) < 4:
        return None, "Detecting beats..."

    peak_times = t[peaks]

    # Avoid reporting stale pulses
    if t[-1] - peak_times[-1] > 1.5:
        return None, "Pulse not detected recently"

    # Inter-beat intervals
    intervals = np.diff(peak_times)

    if np.any((intervals < 0.42) | (intervals > 1.5)):
        return None, "Unstable beats"

    variability = np.std(intervals) / np.mean(intervals)

    if variability > 0.18:
        return None, "Irregular signal"

    bpm = 60 / np.median(intervals)

    return bpm, f"{len(peaks)} peaks | CV={variability:.2f}"


# ====================================
# REAL-TIME LOOP
# ====================================

print("=" * 42)
print("FAST REAL-TIME PPG BPM MONITOR")
print("=" * 42)
print("Place finger on PulseSensor.")
print("Press Ctrl+C to stop.\n")

last_update = time.monotonic()

try:
    while True:

        now = time.monotonic()
        raw = adc.read_raw(CHANNEL)

        samples.append((now, raw))

        while samples and now - samples[0][0] > WINDOW:
            samples.popleft()

        if now - last_update >= UPDATE_INTERVAL:

            bpm, status = calculate_bpm()

            if bpm is not None:
                print(
                    f"BPM: {bpm:.0f} | {status}",
                    flush=True
                )
            else:
                print(
                    f"BPM: -- | {status}",
                    flush=True
                )

            last_update = now

        time.sleep(SAMPLE_INTERVAL)

except KeyboardInterrupt:
    print("\nStopped.")
