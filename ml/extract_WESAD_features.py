"""This file is used to consolidate all the pickle files into one CSV file."""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks

DATA_DIR = Path("data/wesad/pkl")
OUTPUT = Path("data/wesad/wesad_features.csv")

WINDOW_SECONDS = 60

# WESAD wrist sampling rates
RATES = {
    "ACC": 32,
    "BVP": 64,
    "EDA": 4,
    "TEMP": 4
}


def extract_features(file):
    with open(file, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    wrist = data["signal"]["wrist"]
    labels = np.asarray(data["label"]).flatten()

    acc = np.asarray(wrist["ACC"])
    bvp = np.asarray(wrist["BVP"]).flatten()
    eda = np.asarray(wrist["EDA"]).flatten()
    temp = np.asarray(wrist["TEMP"]).flatten()

    rows = []

    # All wrist channels represent the same recording duration.
    window_count = min(
        len(acc) // (32 * WINDOW_SECONDS),
        len(bvp) // (64 * WINDOW_SECONDS),
        len(eda) // (4 * WINDOW_SECONDS),
        len(temp) // (4 * WINDOW_SECONDS)
    )

    for i in range(window_count):
        start = i * WINDOW_SECONDS

        a = acc[i * 32 * 60:(i + 1) * 32 * 60]
        b = bvp[i * 64 * 60:(i + 1) * 64 * 60]
        e = eda[i * 4 * 60:(i + 1) * 4 * 60]
        t = temp[i * 4 * 60:(i + 1) * 4 * 60]

        # BVP pulse peaks; preliminary heart-rate estimate.
        peaks, _ = find_peaks(
            b,
            distance=32,  # minimum 0.5 seconds between peaks
            prominence=np.std(b) * 0.5
        )

        heart_rate = (
            len(peaks) / WINDOW_SECONDS * 60
            if len(peaks) > 1 else np.nan
        )

        # Acceleration magnitude
        acc_magnitude = np.linalg.norm(a, axis=1)

        # Match label samples to this 60-second window.
        label_start = start * 700
        label_end = (start + 60) * 700
        window_labels = labels[label_start:label_end]

        if len(window_labels) == 0:
            continue

        label = int(np.bincount(window_labels).argmax())

        rows.append({
            "subject": file.stem,
            "window_start_seconds": start,
            "heart_rate_bpm": heart_rate,
            "activity_mean": np.mean(acc_magnitude),
            "eda_mean": np.mean(e),
            "eda_std": np.std(e),
            "skin_temp_mean": np.mean(t),
            "skin_temp_std": np.std(t),
            "wesad_label": label
        })

    return rows


all_rows = []

for file in sorted(DATA_DIR.glob("*.pkl")):
    print(f"Processing {file.stem}...")
    all_rows.extend(extract_features(file))

df = pd.DataFrame(all_rows)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("\nExtraction complete!")
print("Shape:", df.shape)
print(df.head())
print("\nMissing values:")
print(df.isna().sum())
print("\nSaved to:", OUTPUT)
