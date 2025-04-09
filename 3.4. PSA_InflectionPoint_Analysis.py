# PSA_InflectionPoint_Analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# === Function: Safe computation of t* ===
def compute_t_star(row, months):
    a = row.values.astype(float)
    if np.all(np.isnan(a)):
        return -1
    try:
        second_derivative = np.gradient(a)
        if np.all(np.isnan(second_derivative)):
            return -1
        idx = np.nanargmax(np.abs(second_derivative))
        return months[idx]
    except Exception:
        return -1

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

# === Step: Compute mean a(t) after t* for each patient ===
def compute_post_t_star_acceleration(df, a_t, months):
    a_post_t_star_list = []

    for idx, row in df.iterrows():
        t_star = row.get("t_star", -1)
        if t_star < 0 or pd.isna(t_star):
            a_post_t_star_list.append(np.nan)
            continue

        if t_star in months:
            t_star_idx = months.index(t_star)
            post_a_values = a_t.iloc[idx, t_star_idx+1:]
            mean_post_a = post_a_values.mean(skipna=True)
            a_post_t_star_list.append(mean_post_a)
        else:
            a_post_t_star_list.append(np.nan)

    df["a_post_t_star"] = a_post_t_star_list
    print("\n📊 Mean a(t) after t* added to DataFrame as 'a_post_t_star'.")
    print(df.groupby("recurrence_flag")["a_post_t_star"].describe())
    return df

# === Step: Plot t* detection and a(t) comparison ===
def plot_t_star_results(df):
    sns.set(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 5)

    plot_df = df[["recurrence_flag", "a_post_t_star", "t_star"]].copy()
    plot_df["group"] = plot_df["recurrence_flag"].map({0: "Non-recurrent", 1: "Recurrent"})

    # Violin plot: a_post_t_star
    plt.figure(figsize=(10, 5))
    sns.violinplot(data=plot_df, x="group", y="a_post_t_star", inner="box", cut=0)
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Post-t* PSA Acceleration by Recurrence Status")
    plt.ylabel("Mean a(t) after t*")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()

    # Bar plot: t* detection rate
    t_star_rate = plot_df.groupby("group")["t_star"].apply(lambda x: (x >= 0).mean() * 100)

    plt.figure(figsize=(6, 5))
    sns.barplot(x=t_star_rate.index, y=t_star_rate.values, palette="pastel")
    plt.ylim(0, 100)
    plt.ylabel("t* Detection Rate (%)")
    plt.title("Structural Inflection Point Detection Rate")
    for i, v in enumerate(t_star_rate.values):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=12)
    plt.tight_layout()
    plt.show()
