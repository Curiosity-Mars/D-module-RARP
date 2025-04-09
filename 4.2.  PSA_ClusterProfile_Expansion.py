# PSA_ClusterProfile_Expansion.py

import pandas as pd
import numpy as np
from scipy.stats import f_oneway, chi2_contingency
from google.colab import drive, files

# === Step 1: Mount Google Drive ===
drive.mount('/content/drive', force_remount=True)

# === Step 2: Load Dataset ===
file_path = '/content/drive/MyDrive/PSA_cluster_results 20250404A.csv'
df = pd.read_csv(file_path)

# === Step 3: Define Extended Variable List ===
extended_vars = [
    'Age', 'Neoadjuvant hormone therapy', 'Height (cm)', 'Weight (kg)', 'BMI',
    'Prostate vol.', 'iPSA', 'Positive Biopsy Ratio (e.g., 2/16 cores = 216)',
    'Biopsy GS A', 'Biopsy GS B', 'Biopsy GS A+B', 'Clinical Stage (cT)', 'Discharge POD',
    'Catheter Removal POD', 'RALP Console Time (min)', 'RARP Surgery Time (min)',
    'RARP Anesthesia Time (min)', 'Intraoperative Blood Loss', 'Pathology GS A',
    'Pathology GS B', 'Pathology GS A+B', 'RM', 'pn', 'sv', 'Pelvic Lymph Dissection (count)'
]

# === Step 4: Identify Variable Types ===
categorical_vars = [v for v in extended_vars if v in df.columns and set(df[v].dropna().unique()) <= {0, 1}]
continuous_vars = [v for v in extended_vars if v in df.columns and v not in categorical_vars]

# === Step 5: Cluster Grouping ===
rows = []
clusters = sorted(df['Cluster'].dropna().unique())

# === Step 6: Summary for Continuous Variables ===
for var in continuous_vars:
    data_by_cluster = [pd.to_numeric(df[df['Cluster'] == c][var], errors='coerce').dropna() for c in clusters]
    means = [v.mean() for v in data_by_cluster]
    stds = [v.std() for v in data_by_cluster]
    try:
        pval = f_oneway(*data_by_cluster).pvalue if all(len(v) > 1 for v in data_by_cluster) else np.nan
    except:
        pval = np.nan
    row = {
        'Category': var,
        **{f'Cluster {c}': f"{means[i]:.2f} ± {stds[i]:.2f}" for i, c in enumerate(clusters)},
        'P-Value': f"{pval:.3f}" if not np.isnan(pval) else ""
    }
    rows.append(row)

# === Step 7: Summary for Categorical Variables ===
for var in categorical_vars:
    contingency = []
    cluster_display = {c: {} for c in clusters}
    for c in clusters:
        sub = df[df['Cluster'] == c][var].dropna()
        count_0 = (sub == 0).sum()
        count_1 = (sub == 1).sum()
        total = count_0 + count_1
        cluster_display[c]['0'] = f"{count_0} ({count_0 / total:.1%})" if total > 0 else "N/A"
        cluster_display[c]['1'] = f"{count_1} ({count_1 / total:.1%})" if total > 0 else "N/A"
        contingency.append([count_0, count_1])
    try:
        pval = chi2_contingency(contingency)[1]
    except:
        pval = np.nan
    rows.append({
        'Category': f"{var} (0)",
        **{f'Cluster {c}': cluster_display[c]['0'] for c in clusters},
        'P-Value': f"{pval:.3f}" if not np.isnan(pval) else ""
    })
    rows.append({
        'Category': f"{var} (1)",
        **{f'Cluster {c}': cluster_display[c]['1'] for c in clusters},
        'P-Value': f"{pval:.3f}" if not np.isnan(pval) else ""
    })

# === Step 8: Add Case Count Per Cluster ===
for c in clusters:
    n = (df['Cluster'] == c).sum()
    rows.insert(0, {'Category': 'Number of cases', f'Cluster {c}': f"{n}"})

# === Step 9: Output Table ===
result_df = pd.DataFrame(rows)
print("\n📊 Cluster-wise Summary Table (Expanded):\n")
print(result_df.to_string(index=False))

# === Step 10: Save and Download CSV ===
result_df.to_csv("expanded_cluster_summary_table.csv", index=False)
files.download("expanded_cluster_summary_table.csv")
