# clusterwise_missing_max_a_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, chi2_contingency
from IPython.display import display

def clusterwise_statistical_comparison(df, continuous_vars, categorical_vars):
    """
    Compare baseline variables within each cluster based on whether max_a is defined or missing.
    Returns: summary DataFrame
    """
    df['max_a_status'] = df['max_a'].apply(lambda x: 'Missing' if pd.isna(x) else 'Defined')
    results = []

    for cluster in sorted(df['Cluster'].dropna().unique()):
        df_c = df[df['Cluster'] == cluster]

        for var in continuous_vars:
            group1 = df_c[df_c['max_a_status'] == 'Missing'][var].dropna()
            group2 = df_c[df_c['max_a_status'] == 'Defined'][var].dropna()
            if len(group1) > 5 and len(group2) > 5:
                tstat, pval = ttest_ind(group1, group2, equal_var=False)
                results.append({
                    'Cluster': cluster,
                    'Variable': var,
                    'Type': 'Continuous',
                    'Missing_mean': group1.mean(),
                    'Defined_mean': group2.mean(),
                    'p_value': pval
                })

        for var in categorical_vars:
            contingency = pd.crosstab(df_c['max_a_status'], df_c[var])
            if contingency.shape[0] == 2 and contingency.shape[1] > 1:
                try:
                    chi2, pval, _, _ = chi2_contingency(contingency)
                    results.append({
                        'Cluster': cluster,
                        'Variable': var,
                        'Type': 'Categorical',
                        'Missing_mean': '–',
                        'Defined_mean': '–',
                        'p_value': pval
                    })
                except:
                    continue

    df_results = pd.DataFrame(results)
    df_results = df_results[['Cluster', 'Variable', 'Type', 'Missing_mean', 'Defined_mean', 'p_value']]
    return df_results.sort_values(['Cluster', 'p_value'])


def plot_psa_trajectories_by_max_a(df, target_cluster=1):
    """
    Plot log(PSA) trajectories in a selected cluster, colored by whether max_a is defined or missing.
    """
    df_cluster = df[df['Cluster'] == target_cluster].copy()
    df_cluster['max_a_status'] = df_cluster['max_a'].apply(lambda x: 'Defined' if pd.notna(x) else 'Missing')

    psa_cols = [col for col in df_cluster.columns if 'PSA POM' in col]
    months = [int(col.replace('PS
