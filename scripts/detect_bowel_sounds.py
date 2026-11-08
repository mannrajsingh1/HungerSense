#!/usr/bin/env python3
"""Print candidate bowel-sound events from a PCM WAV as JSON lines.

This offline bench test analyzes the first channel in 50 ms windows. Its
thresholds are demo defaults and need calibration against annotated recordings.
"""

import argparse
import json
import math
import wave
from itertools import chain


WINDOW_SECONDS = 0.05
LOW_HZ = 100
HIGH_HZ = 1000
MIN_BAND_FRACTION = 0.60


def candidate_windows(recording, min_dbfs):
    sample_rate = recording.getframerate()
    if sample_rate <= 2 * HIGH_HZ:
        raise ValueError("WAV sample rate must be above 2000 Hz")

    width = recording.getsampwidth()
    channel_stride = width * recording.getnchannels()
    full_scale = 1 << (8 * width - 1)
    size = round(sample_rate * WINDOW_SECONDS)
    taper = [0.5 - 0.5 * math.cos(2 * math.pi * i / size) for i in range(size)]
    first_bin = math.ceil(LOW_HZ * size / sample_rate)
    last_bin = math.floor(HIGH_HZ * size / sample_rate)
    coefficients = [
        2 * math.cos(2 * math.pi * k / size)
        for k in range(first_bin, last_bin + 1)
    ]
    min_mean_square = 10 ** (min_dbfs / 10)

    while True:
        start = recording.tell()
        data = recording.readframes(size)
        if len(data) < size * channel_stride:
            break

        if width == 1:
            samples = [(value - 128) / full_scale for value in data[::channel_stride]]
        else:
            samples = [
                int.from_bytes(data[i:i + width], "little", signed=True) / full_scale
                for i in range(0, len(data), channel_stride)
            ]

        mean = sum(samples) / size
        samples = [(value - mean) * weight for value, weight in zip(samples, taper)]
        total_energy = sum(value * value for value in samples)
        if total_energy / size < min_mean_square:
            yield None
            continue

        # Goertzel evaluates only the frequency bins in the band of interest.
        band_power = 0.0
        for coefficient in coefficients:
            previous = older = 0.0
            for value in samples:
                previous, older = value + coefficient * previous - older, previous
            band_power += previous**2 + older**2 - coefficient * previous * older

        band_fraction = 2 * band_power / (size * total_energy)
        yield (start, start + size) if band_fraction >= MIN_BAND_FRACTION else None


def print_events(recording, min_dbfs):
    rate = recording.getframerate()
    event = None
    # A final empty window flushes an event at the end of the recording.
    for window in chain(candidate_windows(recording, min_dbfs), (None,)):
        if window is not None:
            event = (event[0] if event else window[0], window[1])
        elif event is not None:
            print(json.dumps({
                "event": "possible_bowel_sound",
                "start_seconds": round(event[0] / rate, 3),
                "end_seconds": round(event[1] / rate, 3),
            }))
            event = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", help="Path to an uncompressed PCM WAV")
    parser.add_argument("--min-dbfs", type=float, default=-45,
                        help="Minimum windowed RMS level in dBFS (default: -45)")
    args = parser.parse_args()
    if not math.isfinite(args.min_dbfs) or not -120 <= args.min_dbfs <= 0:
        parser.error("--min-dbfs must be between -120 and 0")

    try:
        with wave.open(args.wav, "rb") as recording:
            print_events(recording, args.min_dbfs)
    except (OSError, EOFError, ValueError, wave.Error) as error:
        parser.exit(1, f"Could not analyze WAV: {error}\n")


if __name__ == "__main__":
    main()
