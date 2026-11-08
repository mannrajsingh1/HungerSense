
import time
import statistics
from collections import deque
from grove.adc import ADC

adc = ADC(address=0x04)
CHANNEL = 0

SAMPLE_INTERVAL = 0.01  # 100 samples per second
DISPLAY_INTERVAL = 1.0  # Refresh every second
MIN_BEAT_INTERVAL = 0.35  # Max ~171 BPM
MAX_BEAT_INTERVAL = 1.5   # Min 40 BPM

# Store recent signal readings and heartbeat times
signal = deque(maxlen=100)
beat_times = deque(maxlen=8)

last_beat = None
last_display = 0
was_above = False
smoothed = None

print("Heart Rate Monitor")
print("Place your finger on the sensor")
print("Press Ctrl+C to stop\n")

try:
    while True:
        now = time.monotonic()
        raw = adc.read_raw(CHANNEL)

        # Smooth noisy readings
        if smoothed is None:
            smoothed = float(raw)
        else:
            smoothed = 0.7 * smoothed + 0.3 * raw

        signal.append(smoothed)

        if len(signal) >= 50:
            baseline = statistics.mean(signal)
            amplitude = max(signal) - min(signal)

            # Adaptive threshold based on signal variation
            threshold = baseline + amplitude * 0.30

            above = smoothed > threshold

            # Detect a rising pulse
            if above and not was_above:
                if last_beat is None or (
                    now - last_beat >= MIN_BEAT_INTERVAL
                ):
                    if last_beat is not None:
                        interval = now - last_beat

                        if interval <= MAX_BEAT_INTERVAL:
                            beat_times.append(interval)
                        else:
                            beat_times.clear()

                    last_beat = now

            was_above = above

        # Display once every second
        if now - last_display >= DISPLAY_INTERVAL:
            if last_beat is not None and (
                now - last_beat > MAX_BEAT_INTERVAL * 1.5
            ):
                beat_times.clear()
                last_beat = None

            if len(beat_times) >= 3:
                bpm = 60 / statistics.median(beat_times)
                status = f"{bpm:.0f} BPM"
            else:
                status = "Detecting..."

            print(
                f"\r❤️ Heart Rate: {status:<15} "
                f"| Raw: {raw:<5}",
                end="",
                flush=True
            )

            last_display = now

        time.sleep(SAMPLE_INTERVAL)

except KeyboardInterrupt:
    print("\nMonitoring stopped.")

import time
import statistics
from collections import deque
from grove.adc import ADC

# ---------------- CONFIGURATION ----------------

ADC_ADDRESS = 0x04
CHANNEL = 0

SAMPLE_RATE = 100
DISPLAY_INTERVAL = 1.0

MIN_BPM = 40
MAX_BPM = 180

MIN_AMPLITUDE = 8.0
REQUIRED_INTERVALS = 4
SIGNAL_TIMEOUT = 2.0

# Change to -1 if your sensor's pulse is inverted
POLARITY = 1

# ---------------- HEART RATE MONITOR ----------------

class HeartRateMonitor:
    def __init__(self):
        self.baseline = None
        self.filtered = 0.0

        self.signal = deque()
        self.intervals = deque(maxlen=8)

        self.last_peak = None
        self.tracking_peak = False
        self.peak_value = 0.0
        self.peak_time = None

        self.raw = 0
        self.amplitude = 0.0
        self.quality = "Detecting"

    def reset_beats(self):
        self.intervals.clear()
        self.last_peak = None
        self.tracking_peak = False
        self.peak_value = 0.0
        self.peak_time = None

    def process(self, raw, now):
        self.raw = raw

        # 1. Remove slowly changing baseline
        if self.baseline is None:
            self.baseline = float(raw)
            return

        dt = 1.0 / SAMPLE_RATE

        baseline_alpha = dt / (1.2 + dt)
        self.baseline += baseline_alpha * (
            raw - self.baseline
        )

        detrended = POLARITY * (raw - self.baseline)

        # 2. Smooth high-frequency noise
        smooth_alpha = dt / (0.08 + dt)
        self.filtered += smooth_alpha * (
            detrended - self.filtered
        )

        # 3. Track recent signal amplitude
        self.signal.append((now, self.filtered))

        while self.signal and now - self.signal[0][0] > 3:
            self.signal.popleft()

        if len(self.signal) < SAMPLE_RATE * 2:
            self.quality = "Calibrating"
            return

        values = [value for _, value in self.signal]
        self.amplitude = max(values) - min(values)

        # 4. Reject weak or missing signals
        if self.amplitude < MIN_AMPLITUDE:
            self.quality = "Weak signal"
            self.reset_beats()
            return

        # 5. Adaptive peak threshold
        threshold = max(values) - self.amplitude * 0.65

        # Start tracking a potential heartbeat
        if not self.tracking_peak:
            if self.filtered > threshold:
                self.tracking_peak = True
                self.peak_value = self.filtered
                self.peak_time = now

        else:
            if self.filtered > self.peak_value:
                self.peak_value = self.filtered
                self.peak_time = now

            # Confirm peak after the signal falls
            if self.filtered < threshold:
                self.tracking_peak = False
                self.detect_beat(self.peak_time)

        # 6. Remove stale heartbeat estimates
        if self.last_peak is not None:
            if now - self.last_peak > SIGNAL_TIMEOUT:
                self.reset_beats()
                self.quality = "Searching"

    def detect_beat(self, peak_time):
        if peak_time is None:
            return

        if self.last_peak is None:
            self.last_peak = peak_time
            return

        interval = peak_time - self.last_peak

        min_interval = 60.0 / MAX_BPM
        max_interval = 60.0 / MIN_BPM

        # Reject unrealistic heartbeat timing
        if interval < min_interval:
            return

        if interval > max_interval:
            self.reset_beats()
            self.last_peak = peak_time
            return

        # Reject unusual beat intervals
        if len(self.intervals) >= 3:
            expected = statistics.median(self.intervals)

            if abs(interval - expected) > expected * 0.30:
                return

        self.intervals.append(interval)
        self.last_peak = peak_time

    def get_reading(self, now):
        if len(self.intervals) < REQUIRED_INTERVALS:
            return None, "Detecting"

        if self.last_peak is None:
            return None, "No pulse"

        if now - self.last_peak > SIGNAL_TIMEOUT:
            return None, "No pulse"

        intervals = list(self.intervals)
        median_interval = statistics.median(intervals)

        # Measure beat-to-beat consistency
        deviations = [
            abs(value - median_interval)
            for value in intervals
        ]

        relative_deviation = (
            statistics.median(deviations) / median_interval
        )

        if relative_deviation > 0.15:
            return None, "Unstable"

        bpm = round(60.0 / median_interval)

        return bpm, "Stable"


# ---------------- MAIN PROGRAM ----------------

def main():
    adc = ADC(address=ADC_ADDRESS)
    monitor = HeartRateMonitor()

    sample_interval = 1.0 / SAMPLE_RATE
    last_display = 0.0

    print("\nHEART RATE MONITOR")
    print("----------------------------")
    print("Place finger on sensor")
    print("Keep your hand still")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            start = time.monotonic()

            raw = adc.read_raw(CHANNEL)

            monitor.process(raw, start)

            if start - last_display >= DISPLAY_INTERVAL:
                bpm, status = monitor.get_reading(start)

                bpm_text = (
                    f"{bpm} BPM" if bpm is not None
                    else "-- BPM"
                )

                print(
                    f"\rHeart Rate: {bpm_text:<9}"
                    f" | Signal: {status:<12}"
                    f" | Amplitude: {monitor.amplitude:6.1f}"
                    f" | Raw: {raw:<5}",
                    end="",
                    flush=True
                )

                last_display = start

            elapsed = time.monotonic() - start
            time.sleep(max(0, sample_interval - elapsed))

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


if __name__ == "__main__":
    main()
