# nht_cluster_prediction_and_matched_variable_comparison.py

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from google.colab import files
from io import StringIO, BytesIO
from IPython.display import display

# === PART A: Linear Regression to Predict T_Cluster ===

def predict_nht_cluster():
    print("⬆️ Upload CSV with T_Cluster as target and covariates (e.g., T_Neoadjuvant hormone therapy)")
    uploaded = files.upload()
    filename = next(iter(uploaded))
    df = pd.read_csv(StringIO(uploaded[filename].decode('utf-8')))

    # Example config
    features = ['Age', 'iPSA', 'T_Neoadjuvant hormone therapy']
    target = 'T_Cluster'

    df = df.dropna(subset=features + [target])
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("📈 回帰係数:", model.coef_)
    print("📈 切片:", model.intercept_)
    print(f"🔍 R²: {r2_score(y_test, y_pred):.3f}")
    print(f"🔍 MSE: {mean_squared_error(y_test, y_pred):.3f}")

    df_test = X_test.copy()
    df_test['True Cluster'] = y_test
    df_test['Predicted Cluster'] = y_pred
    print("\n📋 Predicted vs. True Clusters (first 5 rows):")
    display(df_test.head())


# === PART B: Paired Comparison (T_XXX vs C_XXX columns) ===

def compare_matched_variables():
    print("⬆️ Upload paired data with T_XXX and C_XXX columns")
    uploaded = files.upload()
    filename = next(iter(uploaded))
    df = pd.read_csv(BytesIO(uploaded[filename]))

    T_cols = [col for col in df.columns if col.startswith('T_')]
    C_cols = [col for col in df.columns if col.startswith('C_')]
    variables = sorted(set([col[2:] for col in T_cols]) & set([col[2:] for col in C_cols]))

    results = []

    for var in variables:
        t_col, c_col = f"T_{var}", f"C_{var}"
        paired_data = pd.DataFrame({'T': df[t_col], 'C': df[c_col]}).dropna()
        t_vals = paired_data['T']
        c_vals = paired_data['C']

        if pd.api.types.is_numeric_dtype(t_vals) and pd.api.types.is_numeric_dtype(c_vals):
            try:
                t_stat, p_value = stats.ttest_rel(t_vals, c_vals)
            except:
                p_value = np.nan

            results.append([
                var,
                f"{t_vals.mean():.2f} ± {t_vals.std():.2f}",
                f"{c_vals.mean():.2f} ± {c_vals.std():.2f}",
                f"{p_value:.3f}" if pd.notna(p_value) else ""
            ])
        else:
            contingency = pd.crosstab(t_vals, c_vals)
            try:
                if contingency.shape == (2, 2):
                    _, p_value = stats.fisher_exact(contingency)
                else:
                    _, p_value, _, _ = stats.chi2_contingency(contingency)
            except:
                p_value = np.nan

            t_counts = t_vals.value_counts().sort_index()
            c_counts = c_vals.value_counts().sort_index()
            t_str = ", ".join([f"{k} ({v})" for k, v in t_counts.items()])
            c_str = ", ".join([f"{k} ({v})" for k, v in c_counts.items()])
            results.append([var, t_str, c_str, f"{p_value:.3f}" if pd.notna(p_value) else ""])

    summary_df = pd.DataFrame(results, columns=["Category", "NHT", "Non-NHT", "P-Value"])
    print("\n📊 Paired Variable Comparison Table:")
    display(summary_df)
