# paired_group_comparison_nht_vs_non_nht.py

import pandas as pd
import numpy as np
from scipy import stats
from google.colab import files
from io import StringIO
from IPython.display import display

# === 1. Upload CSV file with T_XXX and C_XXX columns ===
uploaded = files.upload()
filename = next(iter(uploaded))
df = pd.read_csv(StringIO(uploaded[filename].decode('utf-8')))

# === 2. Identify matched column pairs ===
t_cols = [col for col in df.columns if col.startswith("T_")]
c_cols = [col for col in df.columns if col.startswith("C_")]
common_labels = [col[2:] for col in t_cols if col[2:] in [c[2:] for c in c_cols]]

# === 3. Initialize result container ===
results = []

# === 4. Compare numeric variables using Welch’s t-test ===
for label in common_labels:
    t_col = "T_" + label
    c_col = "C_" + label

    if np.issubdtype(df[t_col].dtype, np.number) and np.issubdtype(df[c_col].dtype, np.number):
        t_vals = df[t_col].dropna()
        c_vals = df[c_col].dropna()
        t_mean, t_std = t_vals.mean(), t_vals.std()
        c_mean, c_std = c_vals.mean(), c_vals.std()
        _, pval = stats.ttest_ind(t_vals, c_vals, equal_var=False)
        results.append([
            label,
            f"{t_mean:.2f} ± {t_std:.2f}",
            f"{c_mean:.2f} ± {c_std:.2f}",
            f"{pval:.3f}" if pval >= 0.001 else "<0.001"
        ])

# === 5. Compare categorical variables using chi-squared test ===
for label in common_labels:
    t_col = "T_" + label
    c_col = "C_" + label

    if df[t_col].dtype == 'object' or df[c_col].dtype == 'object':
        ct = pd.crosstab(df[t_col], df[c_col])
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            _, pval, _, _ = stats.chi2_contingency(ct)
            for val in ct.index:
                t_count = (df[t_col] == val).sum()
                c_count = (df[c_col] == val).sum()
                total = t_count + c_count
                results.append([
                    f"{label}: {val}",
                    f"{t_count} ({t_count/total:.1%})",
                    f"{c_count} ({c_count/total:.1%})",
                    f"{pval:.3f}" if pval >= 0.001 else "<0.001"
                ])

# === 6. Display results as a table ===
result_df = pd.DataFrame(results, columns=["Category", "NHT", "Non-NHT", "P-Value"])
display(result_df)
