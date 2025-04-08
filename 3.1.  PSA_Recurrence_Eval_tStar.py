# PSA_Recurrence_Eval_tStar.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from google.colab import files

# === Step 1: Upload CSV and load data ===
def upload_and_load_csv():
    uploaded = files.upload()
    filename = next(iter(uploaded))
    try:
        df = pd.read_csv(filename, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filename, encoding='shift-jis')
    return df

# === Step 2: Define columns and preprocess ===
def preprocess_data(df):
    psa_cols = [col for col in df.columns if col.startswith("PSA POM")]
    clinical_cols = [
        'Age', 'BMI', 'Neoadjuvant hormone therapy', 'iPSA',
        'TST', 'LH', 'FSH', 'ACTH', 'Cor', 'Glu',
        'Biopsy GS A+B', 'Pathology GS A+B'
    ]
    df[psa_cols] = df[psa_cols].apply(pd.to_numeric, errors='coerce')
    df[clinical_cols] = df[clinical_cols].apply(pd.to_numeric, errors='coerce')
    return df, psa_cols, clinical_cols

# === Step 3: Define recurrence (2 consecutive PSA ≥ 0.2) ===
def define_recurrence(df, psa_cols):
    def check_recurrence(psa_series, threshold=0.2):
        values = psa_series.values
        count = 0
        for val in values:
            if val >= threshold:
                count += 1
                if count >= 2:
                    return 1
            else:
                count = 0
        return 0
    df['recurrence_flag'] = df[psa_cols].apply(check_recurrence, axis=1)
    return df

# === Step 4: Compute a(t) and t* ===
def compute_t_star(df, psa_cols):
    log_psa = np.log(df[psa_cols].replace(0, np.nan) + 1e-6)
    a_t = -log_psa.diff(axis=1)
    months = [int(col.split("POM")[1]) for col in psa_cols]

    def find_inflection(row):
        a = row.values
        if np.all(np.isnan(a)):
            return -1
        second_derivative = np.gradient(a)
        idx = np.nanargmax(np.abs(second_derivative))
        return months[idx]

    df['t_star'] = a_t.apply(find_inflection, axis=1)
    return df

# === Step 5: Evaluate prediction accuracy using t* ===
def evaluate_prediction(df):
    df['t_star_flag'] = df['t_star'].apply(lambda x: 1 if x >= 0 else 0)
    y_true = df['recurrence_flag']
    y_pred = df['t_star_flag']

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

    return sensitivity, specificity, auc_score
