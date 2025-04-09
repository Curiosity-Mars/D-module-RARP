# ps_matching_variable_sensitivity_analysis.py

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def run_ps_matching_sensitivity(df_encoded, candidate_vars, caliper=0.2, min_size=50):
    """
    Performs sensitivity analysis on all variable combinations used in propensity score (PS) matching.

    Parameters:
        df_encoded (pd.DataFrame): DataFrame with encoded variables including 'Treatment'
        candidate_vars (list): List of candidate variables to consider
        caliper (float): Maximum allowed distance for matching
        min_size (int): Minimum number of patients for a combination to be considered

    Returns:
        pd.DataFrame: Summary of matched pairs by variable combination
    """
    results = []

    for r in range(1, len(candidate_vars) + 1):
        for combo in combinations(candidate_vars, r):
            vars_used = list(combo)
            subset_cols = vars_used + ['Treatment']
            subset_df = df_encoded[subset_cols].dropna()

            if len(subset_df) < min_size:
                continue

            X = subset_df[vars_used]
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

            matched_indices = [i for i, dist in enumerate(distances) if dist[0] <= caliper]
            n_matched = len(matched_indices)

            results.append({
                'Variables': ', '.join(vars_used),
                'Num_Variables': len(vars_used),
                'Num_Matched': n_matched
            })

    return pd.DataFrame(results)


def plot_matching_sensitivity(results_df, save_path_base="/content/ps_matching_sensitivity"):
    """
    Plots the number of matched pairs per variable combination used in PS matching.

    Parameters:
        results_df (pd.DataFrame): DataFrame with columns 'Variables' and 'Num_Matched'
        save_path_base (str): Base path to save .png and .pdf outputs
    """
    labels = results_df['Variables']
    matches = results_df['Num_Matched']
