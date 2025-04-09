# compare_max_a_between_recurrence_groups.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# === Step 1: Compare max_a between recurrence vs. non-recurrence ===

def compare_max_a_statistically(df):
    df_valid = df[df['t_star'].notna()].copy()
    group1 = df_valid[df_valid['recurrence_flag'] == 1]['max_a']
    group0 = df_valid[df_valid['recurrence_flag'] == 0]['max_a']

    stat, p = mannwhitneyu(group1, group0, alternative='greater')

    print("🔍 Comparison of max_a (post-t*) between groups")
    print(f"Median (Recurrence): {group1.median():.3f}")
    print(f"Median (Non-recurrence): {group0.median():.3f}")
    print(f"p-value (one-sided Mann–Whitney U): {p:.4f}")


# === Step 2: Visualize max_a distribution by cluster ===

def plot_max_a_by_cluster(df):
    df_plot = df[['Cluster', 'max_a']].copy()
    df_plot['max_a_label'] = df_plot['max_a'].apply(lambda x: 'Missing' if pd.isna(x) else 'Available')

    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid")
    sns.boxplot(x='Cluster', y='max_a', data=df_plot, showfliers=False)
    sns.stripplot(x='Cluster', y='max_a', data=df_plot[df_plot['max_a_label'] == 'Available'],
                  color='black', alpha=0.3, jitter=0.2, label='Available')

    plt.title("Distribution of max a(t) by Cluster")
    plt.ylabel("max a(t)")
    plt.xlabel("Cluster")
    plt.legend(title='Data Status', loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    missing_counts = df_plot.groupby('Cluster')['max_a'].apply(lambda x: x.isna().sum())
    print("\n🧮 Missing count of max_a per cluster:")
    print(missing_counts)
