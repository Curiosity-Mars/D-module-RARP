# ps_matching_variable_combinations_analysis.py

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def preprocess_dataframe(df, treatment_col='Neoadjuvant hormone therapy', categorical_cols=None):
    """
    Prepares the input dataframe for PS matching analysis.

    Args:
        df (pd.DataFrame): Raw input data
        treatment_col (str): Column name indicating treatment (binary)
        categorical_cols (list): List of categorical variables to encode

    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    df = df.copy()
    df = df.rename(columns={treatment_col: 'Treatment'})
    df['Treatment'] = df['Treatment'].astype(int)

    if categorical_cols:
        for col in categorical_cols:
            df[col] = df[col].astype('category').cat.codes

    return df


def run_ps_matching_combinations(df, candidate_vars, caliper=0.2, min_obs=50):
    """
    Runs PS estimation and nearest neighbor matching for all variable combinations.

    Args:
        df (pd.DataFrame): Preprocessed dataframe including Treatment
        candidate_vars (list): List of candidate covariates
        caliper (float): Matching caliper for nearest neighbor
        min_obs (int): Minimum number of observations to run model

    Returns:
        pd.DataFrame: Summary table of combinations and number of matches
    """
    results = []

    for r in range(1, len(candidate_vars) + 1):
        for combo in combinations(candidate_vars, r):
            vars_used = list(combo)
            subset_cols = vars_used + ['Treatment']
            subset_df = df[subset_cols].dropna().copy()

            if len(subset_df) < min_obs:
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
            if treated.empty or control.empty:
                continue

            nbrs = NearestNeighbors(n_neighbors=1)
            nbrs.fit(control['PS'].values.reshape(-1, 1))
            distances, indices = nbrs.kneighbors(treated['PS'].values.reshape(-1, 1))

            matched_indices = [i for i, dist in enumerate(distances) if dist[0] <= caliper]
            n_matched = len(matched_indices)

            results.append({
                'Variables': ', '.join(vars_used),
                'Num_Variables': len(vars_used),
                'Num_Matched': n_matched
