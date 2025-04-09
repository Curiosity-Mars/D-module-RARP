# compare_max_a_defined_vs_missing.py

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

def compare_max_a_missing_vs_defined(df, continuous_vars, categorical_vars):
    """
    Compare clinical and pathological characteristics between
    patients with and without defined max_a values.

    Parameters:
        df (pd.DataFrame): Input dataframe with clinical columns and 'max_a'
        continuous_vars (list): Column names for t-test
        categorical_vars (list): Column names for chi-squared test

    Returns:
        pd.DataFrame: Summary of group comparisons
    """

    df['max_a_status'] = df['max_a'].apply(lambda x: 'Missing' if pd.isna(x) else 'Defined')
    results = []

    # Continuous variables: t-test
    for var in continuous_vars:
        group1 = df[df['max_a_status'] == 'Missing'][var].dropna()
        group2 = df[df['max_a_status'] == 'Defined'][var].dropna()
        tstat, pval = ttest_ind(group1, group2, equal_var=False)
        results.append({
            'Variable': var,
            'Type': 'Continuous',
            'Missing_mean': group1.mean(),
            'Defined_mean': group2.mean(),
            'p_value': pval
        })

    # Categorical variables: chi-square test
    for var in categorical_vars:
        contingency = pd.crosstab(df['max_a_status'], df[var])
        chi2, pval, _, _ = chi2_contingency(contingency)
        results.append({
            'Variable': var,
            'Type': 'Categorical',
            'Missing_mean': '–',
            'Defined_mean': '–',
            'p_value': pval
        })

    results_df = pd.DataFrame(results)
    results_df = results_df[['Variable', 'Type', 'Missing_mean', 'Defined_mean', 'p_value']]
    return results_df.sort_values('p_value')


# === Example usage (Colab or script) ===
if __name__ == "__main__":
    from google.colab import drive

    # Mount Google Drive and load data
    drive.mount('/content/drive', force_remount=True)
    file_path = '/content/drive/MyDrive/PSA_cluster_results 20250404A.csv'
    df = pd.read_csv(file_path)

    # Define variables
    continuous_vars = ['Age', 'iPSA', 'BMI', 'Prostate vol.']
    categorical_vars = ['Neoadjuvant hormone therapy', 'Biopsy GS A+B', 'Clinical Stage (cT)', 'pT']

    # Run comparison
    results_df = compare_max_a_missing_vs_defined(df, continuous_vars, categorical_vars)

    # Display results
    display(results_df)

    # Optional: save to CSV
    # results_df.to_csv('/content/drive/MyDrive/max_a_missing_vs_defined_results.csv', index=False)
