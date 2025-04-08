# step2_batch_patternA_plotting.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

def extract_psa_columns(df):
    pom_cols = [col for col in df.columns if "PSA POM" in col]
    t_all = np.array([int(col.replace("PSA POM", "")) for col in pom_cols])
    return pom_cols, t_all

def plot_patient_wise(df, output_folder="a_plots", smooth_param=1.0, min_points=4):
    pom_cols, t_all = extract_psa_columns(df)
    os.makedirs(output_folder, exist_ok=True)

    for i, patient in df.iterrows():
        try:
            Y = pd.to_numeric(patient[pom_cols], errors='coerce').astype(float)
            mask = ~np.isnan(Y)
            if np.sum(mask) < min_points:
                continue

            t_valid = t_all[mask]
            Y_valid = Y[mask]

            spline = UnivariateSpline(t_valid, Y_valid, s=smooth_param)
            Y_smooth = spline(t_all)
            dYdt = spline.derivative()(t_all)
            a_t = -dYdt / Y_smooth

            # Plotting
            plt.figure(figsize=(10, 4))

            plt.subplot(1, 2, 1)
            plt.plot(t_all, Y_smooth, label="Smoothed PSA(t)", color="blue")
            plt.scatter(t_valid, Y_valid, color="black", s=15, label="Observed")
            plt.xlabel("Postoperative Months (POM)")
            plt.ylabel("PSA")
            plt.title(f"Patient {i} - PSA(t)")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(t_all, a_t, label="a(t)", color="orange")
            plt.xlabel("Postoperative Months (POM)")
            plt.ylabel("a(t)")
            plt.title(f"Patient {i} - a(t)")
            plt.legend()

            plt.tight_layout()
            plt.savefig(f"{output_folder}/patient_{i}_a_t.png")
            plt.close()

        except Exception as e:
            print(f"Patient {i}: Error - {e}")


# step2_batch_patternB_summary.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import argrelextrema

def extract_psa_columns(df):
    pom_cols = [col for col in df.columns if "PSA POM" in col]
    t_all = np.array([int(col.replace("PSA POM", "")) for col in pom_cols])
    return pom_cols, t_all

def batch_summary_analysis(df, output_folder="a_plots", smooth_param=1.0, min_points=4):
    pom_cols, t_all = extract_psa_columns(df)
    os.makedirs(output_folder, exist_ok=True)

    all_a_t = []
    results = []

    for i, patient in df.iterrows():
        try:
            Y = pd.to_numeric(patient[pom_cols], errors='coerce').astype(float)
            mask = ~np.isnan(Y)
            if np.sum(mask) < min_points:
                continue

            t_valid = t_all[mask]
            Y_valid = Y[mask]
            spline = UnivariateSpline(t_valid, Y_valid, s=smooth_param)
            Y_smooth = spline(t_all)
            dYdt = spline.derivative()(t_all)
            a_t = -dYdt / Y_smooth

            # Inflection point detection
            second_deriv = np.gradient(np.gradient(a_t))
            extrema_idx = argrelextrema(second_deriv, np.greater)[0]
            t_star = t_all[extrema_idx[0]] if len(extrema_idx) > 0 else np.nan

            all_a_t.append(a_t)
            results.append({
                "patient_id": i,
                "inflection_month": t_star,
                "min_a": np.nanmin(a_t)
            })

        except Exception as e:
            print(f"Patient {i}: Error - {e}")

    return np.array(all_a_t), pd.DataFrame(results), t_all

def summarize_a_t(all_a_t, t_all, save_path="figure_a_summary.png"):
    a_mean = np.nanmean(all_a_t, axis=0)
    a_std = np.nanstd(all_a_t, axis=0)

    plt.figure(figsize=(15, 5))

    # A. Example patient
    plt.subplot(1, 3, 1)
    plt.plot(t_all, all_a_t[0], color="blue", label="PSA(t) example")
    plt.xlabel("POM")
    plt.ylabel("PSA")
    plt.title("A. Example PSA trajectory")
    plt.legend()

    # B. All a(t)
    plt.subplot(1, 3, 2)
    for a in all_a_t:
        plt.plot(t_all, a, alpha=0.2, color="orange")
    plt.xlabel("POM")
    plt.ylabel("a(t)")
    plt.title("B. Individual a(t) curves")

    # C. Mean ± SD
    plt.subplot(1, 3, 3)
    plt.plot(t_all, a_mean, color="red", label="Mean a(t)")
    plt.fill_between(t_all, a_mean - a_std, a_mean + a_std, alpha=0.3, color="red")
    plt.xlabel("POM")
    plt.ylabel("a(t)")
    plt.title("C. Mean ± SD of a(t)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
