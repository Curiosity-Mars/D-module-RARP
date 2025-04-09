# ps_matching_dual_model_comparison.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from google.colab import drive


def preprocess_dataframe(file_path, treatment_col='Neoadjuvant hormone therapy', encode_cols=['pT']):
    df = pd.read_csv(file_path)
    df = df.rename(columns={treatment_col: 'Treatment'})
    df['Treatment'] = df['Treatment'].astype(int)

    for col in encode_cols:
        df[col] = df[col].astype('category').cat.codes

    return df


def run_ps_matching(df, covariates, caliper=0.2):
    subset_df = df[covariates + ['Treatment', 't_star', 'max_a']].dropna().copy()
    X = subset_df[covariates]
    y = subset_df['Treatment']

    X_scaled = StandardScaler().fit_transform(X)
    model = LogisticRegression()
    model.fit(X_scaled, y)
    ps = model.predict_proba(X_scaled)[:, 1]
    subset_df['PS'] = ps

    treated = subset_df[subset_df['Treatment'] == 1]
    control = subset_df[subset_df['Treatment'] == 0]

    nbrs = NearestNeighbors(n_neighbors=1)
    nbrs.fit(control['PS'].values.reshape(-1, 1))
    distances, indices = nbrs.kneighbors(treated['PS'].values.reshape(-1, 1))

    matched_treated = []
    matched_control = []

    for i, dist in enumerate(distances):
        if dist[0] <= caliper:
            matched_treated.append(treated.iloc[i])
            matched_control.append(control.iloc[indices[i][0]])

    matched_df = pd.DataFrame(matched_treated).reset_index(drop=True)
    matched_df['Control_t_star'] = [c['t_star'] for c in matched_control]
    matched_df['Control_max_a'] = [c['max_a'] for c in matched_control]

    return matched_df


# === Main Execution ===
if __name__ == "__main__":
    # Mount Google Drive and load data
    drive.mount('/content/drive', force_remount=True)
    file_path = '/content/drive/MyDrive/PSA_cluster_results 20250404A.csv'

    # Preprocess
    df_ps = preprocess_dataframe(file_path, encode_cols=['pT'])

    # Main PS model: Age, iPSA, pT
    matched_df_main = run_ps_matching(df_ps, covariates=["Age", "iPSA", "pT"])

    # Sensitivity model: Age, BMI
    matched_df_sensitivity = run_ps_matching(df_ps, covariates=["Age", "BMI"])

    # Show results
    print("Main Model (Age, iPSA, pT):")
    print(matched_df_main[['t_star', 'Control_t_star', 'max_a', 'Control_max_a']].head())

    print("\nSensitivity
