# PSA_Resilience_ClusterFigures.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from google.colab import files

# === Step 1: Load CSV with cluster and PSA columns ===

def load_clustered_data():
    uploaded = files.upload()
    filename = next(iter(uploaded))
    df = pd.read_csv(filename)

    psa_cols = [col for col in df.columns if col.startswith("PSA POM")]
    df[psa_cols] = df[psa_cols].apply(pd.to_numeric, errors='coerce')

    return df, psa_cols

# === Step 2: Compute a(t) and d^2a/dt^2 ===

def compute_derivatives(df, psa_cols):
    psa_log = np.log(df[psa_cols].replace(0, np.nan))
    a_t = -psa_log.diff(axis=1)
    a_t = a_t.apply(lambda row: gaussian_filter1d(row, sigma=1, mode='nearest'), axis=1, result_type='broadcast')
    a2_t = a_t.apply(lambda row: gaussian_filter1d(np.gradient(row), sigma=1, mode='nearest'), axis=1, result_type='broadcast')
    return a_t, a2_t

# === Step 3: Plot cluster-wise mean a(t) and d^2a/dt^2 ===

def plot_mean_curves(df, a_t, a2_t):
    plt.figure(figsize=(10, 5))
    for c in sorted(df['Cluster'].unique()):
        mean_curve = a_t[df['Cluster'] == c].mean()
        plt.plot(mean_curve.values, label=f"Cluster {c}")
    plt.xlabel("Months After Surgery")
    plt.ylabel("a(t): -d/dt log(PSA)")
    plt.title("Mean a(t) per Cluster")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    for c in sorted(df['Cluster'].unique()):
        mean_curve = a2_t[df['Cluster'] == c].mean()
        plt.plot(mean_curve.values, label=f"Cluster {c}")
    plt.xlabel("Months After Surgery")
    plt.ylabel("d²a/dt²")
    plt.title("Mean Acceleration of a(t) per Cluster")
    plt.legend()
    plt.grid(True)
    plt.show()

# === Step 4: Plot t* histograms by cluster ===

def plot_t_star_histograms(df):
    plt.figure(figsize=(8, 5))
    for c in sorted(df['Cluster'].unique()):
        tstars = df[df['Cluster'] == c]['t_star']
        tstars = tstars[tstars >= 0]
        plt.hist(tstars, bins=20, alpha=0.5, label=f"Cluster {c}")
    plt.xlabel("t* (Months)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Inflection Point t* by Cluster")
    plt.legend()
    plt.grid(True)
    plt.show()

# === Step 5: Print quantitative summary by cluster ===

def summarize_cluster_metrics(df, a_t, a2_t):
    summary_data = []
    for c in sorted(df['Cluster'].unique()):
        idx = df['Cluster'] == c
        t_star_vals = df.loc[idx, 't_star']
        t_star_valid = t_star_vals[t_star_vals >= 0]
        a_vals = a_t.loc[idx]
        a2_vals = a2_t.loc[idx]

        max_a = a_vals.max(axis=1, skipna=True)
        mean_a = a_vals.mean(axis=1, skipna=True)
        max_a2 = a2_vals.max(axis=1, skipna=True)

        summary_data.append({
            'Cluster': c,
            'N': idx.sum(),
            't*_mean': t_star_valid.mean(),
            't*_median': t_star_valid.median(),
            't*_std': t_star_valid.std(),
            't*_valid_ratio': len(t_star_valid) / idx.sum(),
            'max_a_mean': max_a.mean(),
            'mean_a_mean': mean_a.mean(),
            'max_a2_mean': max_a2.mean()
        })

    summary_df = pd.DataFrame(summary_data)
    print("\n📋 Cluster-wise Summary of a(t), t*, and d²a/dt²:\n")
    display(summary_df)
    return summary_df

