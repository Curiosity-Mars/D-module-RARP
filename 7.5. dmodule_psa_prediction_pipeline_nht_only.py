# dmodule_psa_prediction_pipeline_nht_only.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, classification_report
from google.colab import files

# === 1. Upload PS-matched dataset ===
print("⬆️ Upload PS-matched CSV (must include T_ variables)")
uploaded = files.upload()
file_used = [f for f in uploaded if 'PSmatched' in f][0]
df_all = pd.read_csv(file_used)

# === 2. Extract NHT-treated patients (T_ columns only) ===
T_cols = [col for col in df_all.columns if col.startswith('T_')]
df_T = df_all[T_cols].copy()
df_T.columns = [col.replace('T_', '') for col in df_T.columns]
df_T['Neoadjuvant hormone therapy'] = 1
df = df_T.copy()

# === 3. Define features and targets ===
features = ['PSA POM1', 'PSA POM3', 'PSA POM6', 'PSA POM9', 'PSA POM12', 'Age', 'iPSA', 'Neoadjuvant hormone therapy']
targets = ['t_star', 'max_a', 'PSA POM36', 'Cluster']

# Coerce to numeric and drop NaNs
df = df[features + targets].apply(pd.to_numeric, errors='coerce').dropna()
print(f"✅ Valid rows (NHT-only): {len(df)}")

# === 4. Fit regression models ===
X = df[features]

def print_regression_result(name, model, y_true):
    print(f"\n✅ {name} prediction model coefficients:")
    for f, c in zip(features, model.coef_):
        print(f"{f}: {c:.4f}")
    print(f"intercept: {model.intercept_:.4f}")
    print(f"R² = {r2_score(y_true, model.predict(X)):.4f}, MSE = {mean_squared_error(y_true, model.predict(X)):.4f}")

model_tstar = LinearRegression().fit(X, df['t_star'])
print_regression_result("t_star", model_tstar, df['t_star'])

model_maxa = LinearRegression().fit(X, df['max_a'])
print_regression_result("max_a(t)", model_maxa, df['max_a'])

model_psa36 = LinearRegression().fit(X, df['PSA POM36'])
print_regression_result("PSA@36M", model_psa36, df['PSA POM36'])

# === 5. Predict cluster using D-module features ===
X_cluster = pd.DataFrame({
    'pred_t_star': model_tstar.predict(X),
    'pred_max_a': model_maxa.predict(X),
    'pred_psa36': model_psa36.predict(X),
})
y_cluster = df['Cluster']

X_train, X_test, y_train, y_test = train_test_split(X_cluster, y_cluster, test_size=0.3, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n📊 Cluster classification from predicted D-module features:")
print(classification_report(y_test, y_pred))

# === 6. Compare to traditional model (Age + iPSA → PSA@36M) ===
X_trad = df[['Age', 'iPSA']]
y_trad = df['PSA POM36']
model_trad = LinearRegression().fit(X_trad, y_trad)
y_trad_pred = model_trad.predict(X_trad)

print("\n📉 Traditional model (Age + iPSA → PSA@36M):")
print(f"R² = {r2_score(y_trad, y_trad_pred):.4f}")
print(f"MSE = {mean_squared_error(y_trad, y_trad_pred):.4f}")
