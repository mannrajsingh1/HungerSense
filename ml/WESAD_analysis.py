import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import joblib



folder = Path("data/wesad/pkl")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

features = [
    "heart_rate_bpm",
]
FILE_PATH = "data/wesad/wesad_features.csv"
df = pd.read_csv(FILE_PATH)

print(df.shape)
print(df.columns.to_list())
print(df.head())
print(df.describe())


X = df[features].copy()

# Handle missing values
X = X.fillna(X.median())

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans with two clusters
kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X_scaled)

print(df.groupby("cluster")[features].mean())
print(df["cluster"].value_counts())

plt.figure(figsize=(10, 7))

for cluster in sorted(df["cluster"].unique()):
    mask = df["cluster"] == cluster

    plt.scatter(
        df.loc[mask, "window_start_seconds"],
        df.loc[mask, "heart_rate_bpm"],
        label=f"Cluster {cluster}",
        alpha=0.6,
        s=20
    )

plt.xlabel("Time (seconds)")
plt.ylabel("Heart Rate (BPM)")
plt.title("WESAD - Heart Rate Clusters")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


joblib.dump(kmeans, "models/wesad_kmeans.pkl")
joblib.dump(scaler, "models/wesad_scaler.pkl")

print("Model and scaler saved!")