# visualize_max_a_completeness_by_cluster.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_max_a_status_distribution(df):
    """
    Visualize the proportion of defined vs. missing max_a values 
    across clusters, stratified by recurrence status.
    """

    # Label recurrence as 'Yes' or 'No'
    df['recurrence_label'] = df['recurrence_flag'].map({0: 'No', 1: 'Yes'})

    # Define max_a status: 'Defined' or 'Missing'
    df['max_a_status'] = df['max_a'].apply(lambda x: 'Missing' if pd.isna(x) else 'Defined')

    # Group and count
    summary = df.groupby(['Cluster', 'recurrence_label', 'max_a_status']).size().reset_index(name='count')

    # Calculate percentages within each (Cluster × Recurrence) group
    summary['percent'] = summary.groupby(['Cluster', 'recurrence_label'])['count'] \
                                .transform(lambda x: 100 * x / x.sum())

    # Plot
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    sns.barplot(
        data=summary,
        x='Cluster', y='percent', hue='max_a_status',
        ci=None, palette='Set2'
    )
    plt.title("Proportion of max a(t) Status by Cluster and Recurrence")
    plt.ylabel("Percentage (%)")
    plt.xlabel("Cluster")
    plt.legend(title='max a(t) status')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # Display table
    print("📋 Summary of max_a status distribution:")
    print(summary)

    return summary
