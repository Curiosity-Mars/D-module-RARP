# PSA_Compute_tStar_Safe.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# === Function: Safe computation of t* ===
def compute_t_star(row, months):
    """
    Compute the structural inflection point t* from a(t) trajectory.

    Parameters:
    - row: pandas Series representing one patient's a(t)
    - months: list of int, corresponding to month labels (e.g., [1, 2, ..., 120])

    Returns:
    - int: month corresponding to maximum curvature (t*), or -1 if not computable
    """
    a = row.values.astype(float)
    if np.all(np.isnan(a)):
        return -1  # Uncomputable if all values are NaN
    try:
        second_derivative = np.gradient(a)
        if np.all(np.isnan(second_derivative)):
            return -1
        idx = np.nanargmax(np.abs(second_derivative))
        return months[idx]
    except Exception:
        return -1  # Robust to errors (e.g., shape issues)

# === Step: Apply to a_t DataFrame ===
def apply_t_star_and_evaluate(df, a_t, months):
    df["t_star"] = a_t.apply(lambda row: compute_t_star(row, months), axis=1)
    df["t_star_flag"] = df["t_star"].apply(lambda x: 1 if x >= 0 else 0)

    y_true = df["recurrence_flag"]
    y_pred = df["t_star_flag"]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    auc_score = roc_auc_score(y_true, y_pred)

    print(f"✅ Sensitivity: {sensitivity:.2f}")
    print(f"✅ Specificity: {specificity:.2f}")
    print(f"✅ AUC: {auc_score:.2f}")
    print("(Assumption: t* presence predicts recurrence)")

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for t* predicting recurrence")
    plt.grid(True)
    plt.show()

    return df, sensitivity, specificity, auc_score

# === Step: Analyze t* appearance rate in recurrence and non-recurrence groups ===
def analyze_t_star_presence(df):
    df_recur = df[df["recurrence_flag"] == 1]
    df_nonrecur = df[df["recurrence_flag"] == 0]

    recur_with_t_star = (df_recur["t_star"] >= 0).sum()
    recur_total = len(df_recur)
    nonrecur_with_t_star = (df_nonrecur["t_star"] >= 0).sum()
    nonrecur_total = len(df_nonrecur)

    sensitivity = recur_with_t_star / recur_total if recur_total > 0 else np.nan
    false_positive_rate = nonrecur_with_t_star / nonrecur_total if nonrecur_total > 0 else np.nan

    print("\n🎯 t* Appearance Analysis (Structural Breakdown Observation Rate)")
    print(f"Recurrence cases: {recur_total} patients")
    print(f"  ↳ with detected t*: {recur_with_t_star}")
    print(f"  ✅ Sensitivity (t* detection in recurrence): {sensitivity:.2%}")

    print(f"\nNon-recurrence cases: {nonrecur_total} patients")
    print(f"  ↳ with t* falsely detected: {nonrecur_with_t_star}")
    print(f"  ⚠️ False positive rate (t* in non-recurrence): {false_positive_rate:.2%}")
