# PSA_Resilience_Clustering_Ward.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Assumes a_mean, t_all, and all_a_t are already available

def summarize_at_structure(all_a_t, a_mean, t_all):
    min_a_values = []
    t_min_values = []

    for a in all_a_t:
        min_idx = np.nanargmin(a)
        min_a = a[min_idx]
        t_min = t_all[min_idx]
        min_a_values.append(min_a)
        t_min_values.append(t_min)

    mean_min_a = np.mean(min_a_values)
    std_min_a = np.std(min_a_values)
    mean_t_min = np.mean(t_min_values)
    std_t_min = np.std(t_min_values)

    idx_min_avg = np.nanargmin(a_mean)
    a_mean_min = a_mean[idx_min_avg]
    t_mean_min = t_all[idx_min_avg]

    early_idx = (t_all >= 0) & (t_all <= 12)
    late_idx = (t_all >= 60) & (t_all <= 96)
    mean_early = np.nanmean(a_mean[early_idx])
    mean_late = np.nanmean(a_mean[late_idx])

    print("====== a(t) summary statistics ======")
    print(f"Individual min a(t): Mean = {mean_min_a:.3f}, SD = {std_min_a:.3f}")
    print(f"Time of min a(t): Mean = {mean_t_min:.1f} months, SD = {std_t_min:.1f} months")
    print(f"Population mean a(t): Min = {a_mean_min:.3f} at {t_mean_min} months")
    print(f"a(t) average (0–12M): {mean_early:.3f}")
    print(f"a(t) average (60–96M): {mean_late:.3f}")


def cluster_a_t_curves(all_a_t):
    maxlen = max(len(a) for a in all_a_t)
    a_matrix = np.full((len(all_a_t), maxlen), np.nan)

    for i, a in enumerate(all_a_t):
        a_matrix[i, :len(a)] = a

    for i in range(len(a_matrix)):
        for j in range(1, maxlen):
            if np.isnan(a_matrix[i, j]):
                a_matrix[i, j] = a_matrix[i, j-1]

    linkage_matrix = linkage(a_matrix, method='ward')
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix)
    plt.title("Hierarchical Clustering of a(t) curves")
    plt.xlabel("Patient Index")
    plt.ylabel("Distance")
    plt.show()

# Example usage:
# summarize_at_structure(all_a_t, a_mean, t_all)
# cluster_a_t_curves(all_a_t)
