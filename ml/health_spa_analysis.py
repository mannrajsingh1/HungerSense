import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pickle


FILE_PATH = "data/health_spa/health_dataset.csv"
df = pd.read_csv(FILE_PATH)

print(df.shape)
print(df.columns.to_list())
print(df.head())
print(df.describe())


features = ["Heart Rate", "Skin Resistance", "Temperature"]

for feature in features:
    plt.figure(figsize=(7, 4))
    plt.hist(df[feature], bins=30)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.savefig(f"data/health_spa/{feature.replace(' ', '_')}.png")
    plt.close()

    print(f"\n{feature}")
    print("Skewness:", df[feature].skew())
    print("Median:", df[feature].median())
    
    X = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Original Data: ") 
    print(X.head())
    print("Scaled Data:")
    print(X_scaled[:5])
    
    print("\nStandardized means:")
    print(X_scaled.mean(axis=0))

    print("\nStandardized standard deviations:")
    print(X_scaled.std(axis=0))
    
    pca = PCA(n_components = 3)
    X_pca = pca.fit_transform(X_scaled)
    
    
    print("\nExplained variance ratio:")
    print(pca.explained_variance_ratio_)

    print("\nCumulative explained variance:")
    print(pca.explained_variance_ratio_.cumsum())

    print("\nFirst 5 PCA observations:")
    print(X_pca[:5])
    
    
# We use DBSCAN to categorize this data

eps_values = [0.3, 0.5, 0.7, 0.9, 1.0, 1.5]
min_samples_values = [1, 3, 5, 10, 15, 20]

""" 
for eps in eps_values:
    for min_samples in min_samples_values:

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)

        print(
            f"eps={eps}, min_samples={min_samples} | "
            f"Clusters={n_clusters}, Noise={n_noise}"
        )

        if 1 < n_clusters < len(set(labels)):
            non_noise = labels != -1

            score = silhouette_score(
                X_scaled[non_noise],
                labels[non_noise]
            )

            print("Silhouette score:", round(score, 3))
        else:
            print("Silhouette score: Not available")
            
            """
            
dbscan = DBSCAN(eps = 0.5, min_samples = 10)
labels = dbscan.fit_predict(X_scaled)
clustered_data = X.copy()
clustered_data["Cluster"] = labels
print("\nCluster sizes:")
print(clustered_data["Cluster"].value_counts().sort_index())

# Compare average physiological measurements
print("\nAverage measurements per cluster:")
print(
    clustered_data.groupby("Cluster")[features].mean().round(2)
)

# Compare standard deviations
print("\nStandard deviations per cluster:")
print(
    clustered_data.groupby("Cluster")[features].std().round(2)
)


print("\nTemperature statistics by cluster:")
print(
    clustered_data.groupby("Cluster")["Temperature"]
    .describe()
    .round(2)
)

print("\nFeature correlations:")
print(X.corr().round(3))

print(df["Temperature"].head(30).to_list())
print(df["Temperature"].tail(30).to_list())


