# plot_smd_barplot.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files

def plot_smd_barplot(smd_df, save_path='/content/smd_plot.png'):
    """
    Plots a horizontal bar chart of standardized mean differences (SMD) 
    and saves it as a high-resolution PNG.

    Parameters:
        smd_df (pd.DataFrame): DataFrame with at least 'Variable' and 'SMD' columns
        save_path (str): Path to save the figure as PNG (default is for Google Colab)
    """

    plot_df = smd_df.dropna().copy()

    # Plot setup
    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")
    sns.barplot(data=plot_df, y='Variable', x='SMD', palette='coolwarm')

    # Add threshold line for SMD=0.1
    plt.axvline(0.1, color='black', linestyle='--', label='SMD = 0.1 threshold')

    # Labels and title
    plt.title('Standardized Mean Differences (SMD) by Variable', fontsize=14)
    plt.xlabel('Standardized Mean Difference (SMD)', fontsize=12)
    plt.ylabel('Variable', fontsize=12)
    plt.legend()
    plt.tight_layout()

    # Save and display
    plt.savefig(save_path, dpi=300)
    plt.show()

    # Download link (Colab)
    files.download(save_path)
