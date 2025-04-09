# ps_matching_main_vs_sensitivity_export.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from google.colab import drive, files


def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df = df.rename(columns={"Neoadjuvant hormone therapy": "Treatment"})
    df["Treatment"] = df["Treatment"].astype(int)
    df["pT"] = df["pT"].astype("category").cat.codes
    return df


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
    matched_df["Control_t_star"] = [c["t_star"] for c in matched_control]
    matched_df["Control_max_a"] = [c["max_a"] for c in matched_control]
    matched_df["Control_Age"] = [c["Age"] for c in matched_control]
    matched_df["Control_PS"] = [c["PS"] for c in matched_control]

    return matched_df


# === Main Execution ===
if __name__ == "__main__":
    # Mount Drive and load data
    drive.mount('/content/drive', force_remount=True)
    file_path = '/content/drive/MyDrive/PSA_cluster_results 20250404A.csv'
    df_ps = load_and_prepare_data(file_path)

    # Run main and sensitivity matching
    matched_df_main = run_ps_matching(df_ps, covariates=["Age", "iPSA", "pT"])
    matched_df_sensitivity = run_ps_matching(df_ps, covariates=["Age", "BMI"])

    # Export as CSV
    main_output_path = "/content/matched_df_main.csv"
    sensitivity_output_path = "/content/matched_df_sensitivity.csv"

    matched_df_main.to_csv(main_output_path, index=False)
    matched_df_sensitivity.to_csv(sensitivity_output_path, index=False)

    # Download
    files.download(main_output_path)
    files.download(sensitivity_output_path)
