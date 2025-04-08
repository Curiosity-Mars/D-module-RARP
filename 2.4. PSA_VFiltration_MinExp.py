# PSA_VFiltration_MinExp.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from google.colab import files

# === Step 1: Load patient PSA time-series data ===
def load_psa_data():
    uploaded = files.upload()
    filename = next(iter(uploaded))
    df = pd.read_csv(filename)
    psa_cols = [col for col in df.columns if col.startswith("PSA POM")]
    df[psa_cols] = df[psa_cols].apply(pd.to_numeric, errors='coerce')
    return df, psa_cols

# === Step 2: Compute smoothed a(t) ===
def compute_smoothed_at(df, psa_cols):
    psa_log = np.log(df[psa_cols].replace(0, np.nan))
    a_t = -psa_log.diff(axis=1)
    a_t = a_t.apply(lambda row: gaussian_filter1d(row, sigma=1, mode='nearest'), axis=1, result_type='broadcast')
    return a_t

# === Step 3: Estimate minimal exponent proxy from a(t) dynamics ===
def estimate_min_exponents(a_t):
    min_exponents = []
    for idx, row in a_t.iterrows():
        values = row.values
        if np.all(np.isnan(values)):
            min_exponents.append(np.nan)
            continue

        # Proxy: how sharply a(t) falls into negative domain (slope drop point)
        diffs = np.gradient(values)
        if np.any(diffs < -0.05):
            first_idx = np.where(diffs < -0.05)[0][0]
            min_exponents.append(first_idx)
        else:
            min_exponents.append(np.nan)

    return pd.Series(min_exponents, name="min_exponent_index")

# === Step 4: Visualize minimal exponent distribution ===
def plot_min_exponent_distribution(df):
    plt.figure(figsize=(8, 5))
    valid_vals = df['min_exponent_index'].dropna()
    plt.hist(valid_vals, bins=20, color='slateblue', alpha=0.75)
    plt.xlabel("Time Index of First Sharp Decline in a(t)")
    plt.ylabel("Number of Patients")
    plt.title("Distribution of Minimal Exponent Index (V-Filtration Proxy)")
    plt.grid(True)
    plt.show()
