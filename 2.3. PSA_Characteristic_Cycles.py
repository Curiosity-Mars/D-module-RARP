# PSA_Characteristic_Cycles.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from google.colab import files

# === Step 1: Upload and load CSV (must include PSA POM* and t_star) ===
def load_psa_data():
    uploaded = files.upload()
    filename = next(iter(uploaded))
    df = pd.read_csv(filename)
    psa_cols = [col for col in df.columns if col.startswith("PSA POM")]
    df[psa_cols] = df[psa_cols].apply(pd.to_numeric, errors='coerce')
    return df, psa_cols

# === Step 2: Compute a(t) and d²a/dt² ===
def compute_at_characteristics(df, psa_cols):
    psa_log = np.log(df[psa_cols].replace(0, np.nan))
    a_t = -psa_log.diff(axis=1)
    a_t = a_t.apply(lambda row: gaussian_filter1d(row, sigma=1, mode='nearest'), axis=1, result_type='broadcast')
    a2_t = a_t.apply(lambda row: gaussian_filter1d(np.gradient(row), sigma=1, mode='nearest'), axis=1, result_type='broadcast')
    return a_t, a2_t

# === Step 3: Identify peak directions of breakdown (max d²a/dt²) ===
def extract_breakdown_directions(df, a2_t, psa_cols):
    months = [int(col.split("POM")[1]) for col in psa_cols]
    def max_accel_time(row):
        values = row.values
        if np.all(np.isnan(values)):
            return -1
        return months[np.nanargmax(np.abs(values))]
    df['breakdown_time'] = a2_t.apply(max_accel_time, axis=1)
    return df

# === Step 4: Visualize microlocal structure (characteristic cycle proxy) ===
def plot_characteristic_cycles(df):
    plt.figure(figsize=(8, 5))
    valid = df['breakdown_time'] >= 0
    plt.hist(df.loc[valid, 'breakdown_time'], bins=20, color='tomato', alpha=0.7)
    plt.xlabel("Months since surgery")
    plt.ylabel("Number of patients")
    plt.title("Temporal Concentration of Control Breakdown (d²a/dt² peak)")
    plt.grid(True)
    plt.show()
