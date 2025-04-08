# PSA_Irregular_RH_Visualization.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from google.colab import files

# === Step 1: Upload and load CSV (must include PSA POM* and Cluster columns) ===

def load_cluster_data():
    uploaded = files.upload()
    filename = next(iter(uploaded))
    df = pd.read_csv(filename)

    psa_cols = [col for col in df.columns if col.startswith("PSA POM")]
    df[psa_cols] = df[psa_cols].apply(pd.to_numeric, errors='coerce')

    return df, psa_cols

# === Step 2: Compute a(t) and d^2a/dt^2 ===

def compute_at_and_acceleration(df, psa_cols):
    psa_log = np.log(df[psa_cols].replace(0, np.nan))
    a_t = -psa_log.diff(axis=1)
    a_t = a_t.apply(lambda row: gaussian_filter1d(row, sigma=1, mode='nearest'), axis=1, result_type='broadcast')
    a2_t = a_t.apply(lambda row: gaussian_filter1d(np.gradient(row), sigma=1, mode='nearest'), axis=1, result_type='broadcast')
    return a_t, a2_t

# === Step 3: Plot mean a(t) by cluster ===

def plot_mean_at(df, a_t):
    plt.figure(figsize=(10, 5))
    for c in sorted(df['Cluster'].unique()):
        group = a_t[df['Cluster'] == c]
        mean_curve = group.mean()
        plt.plot(mean_curve.values, label=f"Cluster {c}")
    plt.xlabel("Months After Surgery")
    plt.ylabel("a(t): -d/dt log(PSA)")
    plt.title("Mean a(t) per Cluster")
    plt.legend()
    plt.grid(True)
    plt.show()

# === Step 4: Plot mean d^2a/dt^2 by cluster ===

def plot_mean_acceleration(df, a2_t):
    plt.figure(figsize=(10, 5))
    for c in sorted(df['Cluster'].unique()):
        group = a2_t[df['Cluster'] == c]
        mean_curve = group.mean()
        plt.plot(mean_curve.values, label=f"Cluster {c}")
    plt.xlabel("Months After Surgery")
    plt.ylabel("d²a/dt²")
    plt.title("Mean Acceleration of a(t) per Cluster")
    plt.legend()
    plt.grid(True)
    plt.show()

# === Step 5: Plot t* histogram by cluster ===

def plot_t_star_distribution(df):
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
