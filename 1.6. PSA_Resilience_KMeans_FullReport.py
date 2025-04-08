# PSA_Resilience_StructureClustering.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import kmapper as km
import umap
from sklearn import cluster

# Assumes a_mean, t_all, all_a_t, df, and pom_cols are available

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


def cluster_a_t_curves(all_a_t, k=4):
    maxlen = max(len(a) for a in all_a_t)
    a_matrix = np.full((len(all_a_t), maxlen), np.nan)

    for i, a in enumerate(all_a_t):
        a_matrix[i, :len(a)] = a

    for i in range(len(a_matrix)):
        for j in range(1, maxlen):
            if np.isnan(a_matrix[i, j]):
                a_matrix[i, j] = a_matrix[i, j-1]

    linkage_matrix = linkage(a_matrix, method='ward')
    cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust')

    # Plot cluster-wise mean curves
    plt.figure(figsize=(10, 6))
    for c in range(1, k + 1):
        cluster_members = a_matrix[cluster_labels == c]
        mean_curve = np.nanmean(cluster_members, axis=0)
        plt.plot(mean_curve, label=f"Cluster {c}")
    plt.title("Mean a(t) curves by cluster (Ward)")
    plt.xlabel("Time index (months)")
    plt.ylabel("a(t)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return cluster_labels, a_matrix


def export_cluster_means(a_matrix, cluster_labels, k=4):
    cluster_mean_dict = {"month": np.arange(a_matrix.shape[1])}
    for c in range(1, k + 1):
        cluster_members = a_matrix[cluster_labels == c]
        mean_curve = np.nanmean(cluster_members, axis=0)
        cluster_mean_dict[f"Cluster_{c}"] = mean_curve

    df_cluster_means = pd.DataFrame(cluster_mean_dict)
    df_cluster_means.to_csv("cluster_mean_a_t.csv", index=False)
    print("Saved: cluster_mean_a_t.csv")


def assign_clusters_to_df(df, cluster_labels, pom_cols):
    valid_indices = []
    for i, patient in df.iterrows():
        Y = patient[pom_cols].values
        Y = pd.to_numeric(Y, errors='coerce').astype(float)
        if np.sum(~np.isnan(Y)) >= 4:
            valid_indices.append(i)

    df_clustered = df.iloc[valid_indices].copy()
    df_clustered["cluster"] = cluster_labels
    df_clustered.to_csv("patient_with_clusters.csv", index=False)
    print("Saved: patient_with_clusters.csv")
    return df_clustered


def run_mapper_analysis(a_matrix, df_clustered):
    a_matrix_clean = pd.DataFrame(a_matrix).fillna(method='ffill', axis=1).fillna(0).values

    mapper = km.KeplerMapper(verbose=1)
    projected = umap.UMAP(n_components=2, random_state=42).fit_transform(a_matrix_clean)

    graph = mapper.map(projected, a_matrix_clean,
                       cover=km.Cover(n_cubes=15, perc_overlap=0.5),
                       clusterer=cluster.DBSCAN(eps=0.5, min_samples=5))

    mapper.visualize(graph,
                     path_html="tda_mapper_a_t.html",
                     title="TDA Mapper: a(t) Resilience Landscape")

    rows = []
    for node_key, patient_indices in graph["nodes"].items():
        for pid in patient_indices:
            row = {
                "node_id": node_key,
                "patient_index": pid,
            }
            patient_info = df_clustered.iloc[pid].to_dict()
            row.update(patient_info)
            rows.append(row)

    df_mapper_nodes = pd.DataFrame(rows)
    df_mapper_nodes.to_csv("mapper_nodes_with_patients.csv", index=False)
    print("Saved: mapper_nodes_with_patients.csv")
    return df_mapper_nodes

# Example usage:
# summarize_at_structure(all_a_t, a_mean, t_all)
# cluster_labels, a_matrix = cluster_a_t_curves(all_a_t, k=4)
# export_cluster_means(a_matrix, cluster_labels, k=4)
# df_clustered = assign_clusters_to_df(df, cluster_labels, pom_cols)
# df_mapper_nodes = run_mapper_analysis(a_matrix, df_clustered)
