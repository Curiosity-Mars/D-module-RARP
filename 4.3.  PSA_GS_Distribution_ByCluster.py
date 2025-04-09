# PSA_GS_Distribution_ByCluster.py

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# === Target variable and clustering ===
target_var = "Biopsy GS A+B"
clusters = sorted(df['Cluster'].dropna().unique())

# === Step 1: Clean and extract valid rows ===
data = df[[target_var, 'Cluster']].dropna()
score_levels = sorted(data[target_var].unique())

# === Step 2: Cross-tabulation and percent formatting ===
rows = []
contingency = []

for score in score_levels:
    row = {'Biopsy GS A+B': int(score) if not isinstance(score, str) else score}
    count_row = []
    for c in clusters:
        group = data[(data['Cluster'] == c) & (data[target_var] == score)]
        total = (df['Cluster'] == c).sum()
        count = len(group)
        percent = count / total * 100 if total > 0 else 0
        row[f'Cluster {c}'] = f"{count} ({percent:.1f}%)"
        count_row.append(count)
    rows.append(row)
    contingency.append(count_row)

# === Step 3: Chi-square test ===
try:
    chi2, pval, _, _ = chi2_contingency(contingency)
    pval_str = f"{pval:.3f}"
except:
    pval_str = ""

for row in rows:
    row["P-Value"] = pval_str

# === Step 4: Output table ===
gs_table = pd.DataFrame(rows)
print("\n📊 Distribution of Biopsy GS A+B by Cluster:\n")
print(gs_table.to_string(index=False))
