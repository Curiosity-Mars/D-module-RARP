# dmodule_psa_validation_pipeline.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, classification_report
from google.colab import files
from io import StringIO

# === 1. Upload Files ===
print("⬆️ Upload:")
print(" - Full dataset (e.g., PSA_cluster_results.csv)")
print(" - Used patients file (e.g., PSmatched.csv)")

uploaded = files.upload()
file_all = [f for f in uploaded.keys() if 'cluster_results' in f or 'PSA_cluster_results' in f][0]
file_used = [f for f in uploaded.keys() if 'PSmatched' in f][0]

# === 2. Load Data ===
all_df = pd.read_csv(file_all)
used_df = pd.read_csv(file_used)

# === 3. Extract validation data (unused patients) ===
used_ids = set(used_df['Patient_ID'].dropna().astype(int))
unused_df = all_df[~all_df['Patient_ID'].isin(used_ids)].copy()
print(f"\n✅ Validation dataset: {unused_df.shape[0]} rows (before filtering)")

# === 4. Select and clean relevant columns ===
cols_required = [
    'Patient_ID', 't_star', 'max_a',
    'PSA POM1', 'PSA POM3', 'PSA POM6', 'PSA POM9', 'PSA POM12', 'PSA POM36',
    'Age', 'iPSA', 'Cluster'
]
unused_df = unused_df[unused_df.columns.intersection(cols_required)].dropna()
print(f"📉 After dropna: {unused_df.shape[0]} rows")

# === 5. Predict structural indicators from early PSA and clinical features ===
X_test = unused_df[['PSA POM1', 'PSA POM3', 'PSA POM6', 'PSA POM9', 'PSA POM12', 'Age', 'iPSA']]

def fit_and_predict(target_name):
    y = unused_df[target_name]
    model = LinearRegression().fit(X_test, y)
    return model.predict(X_test)

unused_df['pred_t_star'] = fit_and_predict('t_star')
unused_df['pred_max_a'] = fit_and_predict('max_a')
unused_df['pred_psa36'] = fit_and_predict('PSA POM36')

# === 6. Predict cluster using D-module outputs ===
cluster_X = unused_df[['pred_t_star', 'pred_max_a', 'pred_psa36']]
cluster_y = unused_df['Cluster']
clf = RandomForestClassifier(random_state=42).fit(cluster_X, cluster_y)
cluster_pred = clf.predict(cluster_X)

print("\n📊 Classification Report (D-module predicted cluster):")
print(classification_report(cluster_y, cluster_pred))

# === 7. Benchmark: Traditional linear model to predict PSA@36M ===
X_traditional = unused_df[['Age', 'iPSA']]
y_traditional = unused_df['PSA POM36']
model_traditional = LinearRegression().fit(X_traditional, y_traditional)
pred_traditional = model_traditional.predict(X_traditional)

r2 = r2_score(y_traditional, pred_traditional)
mse = mean_squared_error(y_traditional, pred_traditional)
print("\n📉 Baseline Model (Age + iPSA → PSA@36M):")
print(f"R² = {r2:.4f}")
print(f"MSE = {mse:.4f}")
