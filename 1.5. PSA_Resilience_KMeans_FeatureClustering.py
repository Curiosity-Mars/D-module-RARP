# PSA_Resilience_KMeans_FeatureClustering.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import kmapper as km
import umap
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from scipy.ndimage import gaussian_filter1d
from google.colab import files

# New step: Feature-based clustering from raw PSA data

def cluster_from_psa_file():
    uploaded = files.upload()
    filename = next(iter(uploaded))
    df = pd.read_csv(filename)

    # Extract PSA POM1–POM120 columns
    psa_cols = [col for col in df.columns if col.startswith("PSA POM") and col != "PSA POM0"]
    psa_data = df[psa_cols].copy()
    psa_data = psa_data.apply(pd.to_numeric, errors='coerce')

    # Log-transform and compute smoothed derivative
    psa_log = np.log(psa_data.replace(0, np.nan))
    a_t = psa_log.diff(axis=1) / 1.0
    a_t = a_t.apply(lambda row: gaussian_filter1d(row, sigma=1, mode='nearest'), axis=1, result_type='broadcast')

    # Extract time axis (POM in months)
    pom_months = [int(col.split("POM")[1]) for col in psa_cols]

    # Feature extraction
    features = pd.DataFrame(index=df.index)
    features['max_a'] = a_t.max(axis=1, skipna=True)
    features['mean_a'] = a_t.mean(axis=1, skipna=True)
    features['first_rise_month'] = a_t.apply(
        lambda row: pom_months[np.argmax(row > 0)] if np.any(row > 0) else -1,
        axis=1
    )

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(features)

    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)
    df['cluster'] = clusters

    # Plot mean PSA curves by cluster
    for c in sorted(df['cluster'].unique()):
        group = psa_data[df['cluster'] == c]
        mean_curve = group.mean(skipna=True)
        plt.plot(mean_curve.values, label=f"Cluster {c}")

    plt.xlabel("Months After Surgery")
    plt.ylabel("PSA (mean per cluster)")
    plt.title("Mean PSA Trajectories by Cluster")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print cluster sizes
    print("Cluster sizes:")
    cluster_counts = df['cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count} patients")

    return df, psa_data, features, a_t
