# dmodule_validation_pipeline_colab.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, classification_report
from google.colab import files
from io import BytesIO

# === 1. Upload two CSV files: full dataset + training-used patients ===
print("⬆️ Upload:")
print(" - Full data (e.g., PSA_cluster_results_*.csv)")
print(" - Used patients file (e.g., PSmatched_*.csv)")

uploaded = files.upload()
file_all = [f for f in uploaded.keys() if 'cluster_results' in f][0]
file_used = [f for f in uploaded.keys() if 'PSmatched' in f][0]

all_df = pd.read_csv(BytesIO(uploaded[file_all]))
used_df = pd.read_csv(BytesIO(uploaded[file_used]))

# === 2. Extract unused validation patients ===
used_ids = set(used_df['Patient_ID'].dropna().astype(int))
val_df = all_df[~all_df['Patient_ID'].isin(used_ids)].copy()
print(f"\n✅ Validation set size before dropna: {len(val_df)}")

# === 3. Select and clean columns ===
cols_needed = [
    'PSA POM1', 'PSA POM3', 'PSA POM6', 'PSA POM9', 'PSA POM12',
    'Age', 'iPSA', 'Neoadjuvant hormone therapy',
    't_star', 'max_a', 'PSA POM36', 'Cluster'
]
val_df = val_df[cols_needed].apply(pd.to_numeric, errors='coerce').dropna()
print(f"📉 After dropna: {len(val_df)} rows")

if val_df.empty:
    raise ValueError("❌ No rows remain after filtering. Check for missing data.")

# === 4. Define inputs ===
X_val = val_df[['PSA POM1','PSA POM3','PSA POM6','PSA POM9','PSA POM12','Age','iPSA','Neoadjuvant hormone therapy']]

# === 5. Train simple linear regression models ===
def train_model(X, y):
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    return model, pred, r2_score(y, pred), mean_squared_error(y, pred)

_, pred_t, _, _ = train_model(X_val, val_df['t_star'])
_, pred_a, _, _ = train_model(X_val, val_df['max_a'])
_, pred_p, _, _ = train_model(X_val, val_df['PSA POM36'])

val_df['pred_t_star'] = pred_t
val_df['pred_max_a'] = pred_a
val_df['pred_psa36'] = pred_p

# === 6. Cluster prediction using D-module features ===
X_cls = val_df[['pred_t_star', 'pred_max_a', 'pred_psa36']]
y_cls = val_df['Cluster']
clf = RandomForestClassifier(random_state=42).fit(X_cls, y_cls)
y_pred = clf.predict(X_cls)

print("\n📊 Classification performance using D-module predictions:")
print(classification_report(y_cls, y_pred))

# === 7. Baseline model: Age + iPSA + NHT → PSA@36M ===
X_baseline = val_df[['Age','iPSA','Neoadjuvant hormone therapy']]
y_baseline = val_df['PSA POM36']
baseline_model = LinearRegression().fit(X_baseline, y_baseline)
baseline_pred = baseline_model.predict(X_baseline)
print("\n📉 Baseline model (Age+iPSA+NHT → PSA@36M):")
print(f"R² = {r2_score(y_baseline, baseline_pred):.4f}")
print(f"MSE = {mean_squared_error(y_baseline, baseline_pred):.4f}")
