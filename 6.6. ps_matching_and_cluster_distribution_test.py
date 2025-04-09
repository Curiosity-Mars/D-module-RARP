# ps_matching_and_cluster_distribution_test.py

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from google.colab import drive, files, data_table


def run_ps_matching(df, covariates, caliper=0.2):
    subset_df = df[covariates + ['Patient_ID', 'Age', 'Treatment', 't_star', 'max_a']].dropna().copy()
    X = subset_df[covariates]
    y = subset_df['Treatment']

    X_scaled = StandardScaler().fit_transform(X)
    model = LogisticRegression()
    model.fit(X_scaled, y)
    ps = model.predict_proba(X_scaled)[:, 1]
    subset_df["PS"] = ps

    treated = subset_df[subset_df["Treatment"] == 1]
    control = subset_df[subset_df["Treatment"] == 0]

    nbrs = NearestNeighbors(n_neighbors=1)
    nbrs.fit(control["PS"].values.reshape(-1, 1))
    distances, indices = nbrs.kneighbors(treated["PS"].values.reshape(-1, 1))

    matched_treated = []
    matched_control = []

    for i, dist in enumerate(distances):
        if dist[0] <= caliper:
            matched_treated.append(treated.iloc[i].to_dict())
            matched_control.append(control.iloc[indices[i][0]].to_dict())

    matched_df = pd.DataFrame(matched_treated).reset_index(drop=True)
    matched_df["Control_Patient_ID"] = [c["Patient_ID"] for c in matched_control]
    return matched_df


def compare_cluster_distribution(matched_df, full_df):
    ids_nht = matched_df["Patient_ID"].tolist()
    ids_non_nht = matched_df["Control_Patient_ID"].tolist()

    df_nht = full_df[full_df["Patient_ID"].isin(ids_nht)].copy()
    df_nht["Group"] = "NHT"

    df_non_nht = full_df[full_df["Patient_ID"].isin(ids_non_nht)].copy()
    df_non_nht["Group"] = "No NHT"

    df_compare = pd.concat([df_nht, df_non_nht], ignore_index=True)

    cluster_table = pd.crosstab(df_compare["Cluster"], df_compare["Group"])
    cluster_percent = cluster_table.div(cluster_table.sum(axis=1), axis=0) * 100
    chi2, p_val, dof, expected = chi2_contingency(cluster_table)

    result_rows = []
    for cluster in cluster_table.index:
        count_nht = cluster_table.loc[cluster, "NHT"]
        count_non_nht = cluster_table.loc[cluster, "No NHT"]
        percent_nht = cluster_percent.loc[cluster, "NHT"]
        percent_non_nht = cluster_percent.loc[cluster, "No NHT"]
        result_rows.append({
            "Cluster": f"Cluster {cluster}",
            "NHT Group": f"{count_nht} ({percent_nht:.1f}%)",
            "No NHT Group": f"{count_non_nht} ({percent_non_nht:.1f}%)",
            "P-Value": f"{p_val:.3f}"
        })

    df_summary = pd.DataFrame(result_rows)
    return df_summary


# === Main Execution ===
if __name__ == "__main__":
    # 1. Load main dataset
    drive.mount('/content/drive', force_remount=True)
    file_path = '/content/drive/MyDrive/PSA_cluster_results 20250404A.csv'
    df = pd.read_csv(file_path)
    df = df.rename(columns={"Neoadjuvant hormone therapy": "Treatment"})
    df["Treatment"] = df["Treatment"].astype(int)

    # 2. Run matching (e.g., using Age and BMI)
    matched_df = run_ps_matching(df, covariates=["Age", "BMI"])

    # 3. Save matched results
    matched_path = "/content/matched_df_sensitivity_Age_BMI.csv"
    matched_df.to_csv(matched_path, index=False)
    files.download(matched_path)

    # 4. Run cluster distribution comparison
    summary_df = compare_cluster_distribution(matched_df, df)

    # 5. Show table (interactive)
    data_table.DataTable(summary_df)
