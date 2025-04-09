# visualize_max_a_and_t_star_completeness.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_log_psa_trajectories_by_max_a(df, target_cluster=1):
    df_c = df[df['Cluster'] == target_cluster].copy()
    df_c['max_a_status'] = df_c['max_a'].apply(lambda x: 'Defined' if pd.notna(x) else 'Missing')

    psa_cols = [col for col in df_c.columns if 'PSA POM' in col]
    months = [int(col.replace('PSA POM', '')) for col in psa_cols]

    color_map = {'Defined': 'red', 'Missing': 'gray'}
    alpha_map = {'Defined': 0.6, 'Missing': 0.2}

    plt.figure(figsize=(10, 6))
    for _, row in df_c.iterrows():
        psa_values = pd.to_numeric(row[psa_cols], errors='coerce').values
        if np.isnan(psa_values).all():
            continue
        log_psa = np.log10(psa_values + 1e-4)
        label = row['max_a_status']
        plt.plot(months, log_psa, color=color_map[label], alpha=alpha_map[label])

    plt.title(f'Log(PSA) Trajectories in Cluster {target_cluster}\nColored by max a(t) Definition Status')
    plt.xlabel('Months After Surgery')
    plt.ylabel('log10(PSA + 0.0001)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_t_star_distribution_and_missing_rate(df):
    df['t_star_status'] = df['t_star'].apply(lambda x: 'Missing' if pd.isna(x) else 'Defined')
    df_valid = df[df['t_star_status'] == 'Defined'].copy()

    missing_summary = df.groupby('Cluster')['t_star'].apply(lambda x: x.isna().mean()).reset_index()
    missing_summary.columns = ['Cluster', 'MissingRate']
    missing_summary['MissingRate'] = missing_summary['MissingRate'].astype(float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)

    sns.violinplot(data=df_valid, x='Cluster', y='t_star', ax=ax1, palette='pastel', cut=0, inner='quartile')
    ax1.set_title("Distribution of t* (Inflection Point) by Cluster")
    ax1.set_ylabel("t* (months)")
    ax1.grid(True)

    sns.barplot(data=missing_summary, x='Cluster', y='MissingRate', ax=ax2, color='gray')
    ax2.set_ylabel("Missing Rate")
    ax2.set_ylim(0, 1)
    ax2.set_title("Proportion of Missing t* by Cluster")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.grid(axis='y')

    plt.xlabel("Cluster")
    plt.tight_layout()
    plt.show()


def plot_max_a_definition_vs_followup(df):
    psa_cols = [col for col in df.columns if 'PSA POM' in col]
    months = [int(col.replace('PSA POM', '')) for col in psa_cols]

    def get_last_month(row):
        for col, month in zip(reversed(psa_cols), reversed(months)):
            try:
                val = float(row[col])
                if not np.isnan(val):
                    return month
            except:
                continue
        return np.nan

    df['last_psa_month'] = df.apply(get_last_month, axis=1)
    df['max_a_defined'] = df['max_a'].apply(lambda x: 1 if pd.notna(x) else 0)

    if 'recurrence_flag' in df.columns:
        df['recurrence_label'] = df['recurrence_flag'].map({0: 'No', 1: 'Yes'})
    else:
        df['recurrence_label'] = 'Unknown'

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='last_psa_month',
        y='max_a_defined',
        hue='recurrence_label',
        palette='Set1',
        alpha=0.6
    )

    plt.title("Definition of max a(t) vs. Last Follow-up Month")
    plt.xlabel("Last Month with PSA Observation")
    plt.ylabel("max a(t) Defined (1) or Missing (0)")
    plt.yticks([0, 1], ['Missing', 'Defined'])
    plt.grid(True)
    plt.tight_layout()
    plt.show()
