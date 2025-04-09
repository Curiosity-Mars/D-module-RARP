# PSA_TStar_Recurrence_Eval.py

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from google.colab import drive

# === Step 1: Prepare DataFrame with 't_star_valid' and 'recurrence_flag' ===
df_filtered = df.dropna(subset=['t_star_valid', 'recurrence_flag'])

# === Step 2: Cross-tabulate recurrence and t* detection ===
contingency = pd.crosstab(df_filtered['recurrence_flag'], df_filtered['t_star_valid'])
print("\n🔢 Cross-tabulation (Recurrence vs t* Detection):")
display(contingency)

# === Step 3: Extract TP, FN, FP, TN ===
TP = contingency.loc[1, True] if (1 in contingency.index and True in contingency.columns) else 0
FN = contingency.loc[1, False] if (1 in contingency.index and False in contingency.columns) else 0
FP = contingency.loc[0, True] if (0 in contingency.index and True in contingency.columns) else 0
TN = contingency.loc[0, False] if (0 in contingency.index and False in contingency.columns) else 0

# === Step 4: Calculate metrics ===
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else np.nan
specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan
ppv = TP / (TP + FP) if (TP + FP) > 0 else np.nan
npv = TN / (TN + FN) if (TN + FN) > 0 else np.nan

print("\n📊 Performance Metrics:")
print(f"Sensitivity: {sensitivity:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"Positive Predictive Value (PPV): {ppv:.3f}")
print(f"Negative Predictive Value (NPV): {npv:.3f}")

# === Step 5: Chi-square test for independence ===
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"\n📈 Chi-square test p-value: {p:.4f}")
