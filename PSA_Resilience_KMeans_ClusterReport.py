# PSA_Resilience_KMeans_ClusterReport.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import silhouette_score
from google.colab import files

# === Step 1: Upload and preprocess PSA data ===
def cluster_from_psa_file():
    uploaded = files.upload()
    filename = next(iter(uploaded))
    df = pd.read_csv(filename)

    psa_cols = [col for col in df.columns if col.startswith("PSA POM") and col != "PSA POM0"]
    psa_data = df[psa_cols].copy().apply(pd.to_numeric, errors='coerce')
    psa_log = np.log(psa_data.replace(0, np.nan))
    a_t = psa_log.diff(axis=1) / 1.0
    a_t = a_t.apply(lambda row: gaussian_filter1d(row, sigma=1, mode='nearest'), axis=1, result_type='broadcast')

    pom_months = [int(col.split("POM")[1]) for col in psa_cols]

    features = pd.DataFrame(index=df.index)
    features['max_a'] = a_t.max(axis=1, skipna=True)
    features['mean_a'] = a_t.mean(axis=1, skipna=True)
    features['first_rise_month'] = a_t.apply(
        lambda row: pom_months[np.argmax(row > 0)] if np.any(row > 0) else -1,
        axis=1
    )

    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)
    df['cluster'] = clusters

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

    print("Cluster sizes:")
    cluster_counts = df['cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count} patients")

    return df, psa_data, features, a_t, psa_cols, pom_months


# === Step 2: Compute additional structural indicators and export ===
def add_structural_indicators(df, features, psa_data, a_t, pom_months):
    def compute_inflection(row):
        first_derivative = row.values
        second_derivative = np.gradient(first_derivative)
        if np.all(np.isnan(second_derivative)):
            return -1
        peak_idx = np.nanargmax(np.abs(second_derivative))
        return pom_months[peak_idx]

    features['t_star'] = a_t.apply(compute_inflection, axis=1)
    features['psa_peak_month'] = psa_data.apply(
        lambda row: pom_months[np.nanargmax(row.values)] if row.notna().any() else -1,
        axis=1)

    cluster_output = pd.DataFrame({
        'Patient_ID': df.index,
        'Cluster': df['cluster'],
        't_star': features['t_star'],
        'first_rise_month': features['first_rise_month'],
        'psa_peak_month': features['psa_peak_month'],
        'max_a': features['max_a'],
        'mean_a': features['mean_a']
    })

    cluster_output.to_csv("PSA_cluster_results.csv", index=False)
    files.download("PSA_cluster_results.csv")
    return cluster_output


# === Step 3: Silhouette Score Evaluation ===
def evaluate_cluster_number(features):
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(features)

    sil_scores = []
    K_range = range(2, 7)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        sil_scores.append(score)

    plt.figure(figsize=(6, 4))
    plt.plot(K_range, sil_scores, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.grid(True)
    plt.xticks(K_range)
    plt.show()

    best_k = K_range[np.argmax(sil_scores)]
    print(f"✅ Best K by Silhouette Score: {best_k} (Score = {max(sil_scores):.3f})")
    return best_k


# === Step 4: Elbow Method Evaluation ===
def elbow_method(features):
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(features)

    sse = []
    K_range = range(1, 11)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(6, 4))
    plt.plot(K_range, sse, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('Elbow Method for Optimal K')
    plt.grid(True)
    plt.xticks(K_range)
    plt.show()
