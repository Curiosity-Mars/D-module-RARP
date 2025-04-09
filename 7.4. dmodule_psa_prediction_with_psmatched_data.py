# dmodule_psa_prediction_with_psmatched_data.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from google.colab import files

# === 1. Upload PS-matched dataset ===
print("⬆️ Upload PS-matched dataset (e.g., PSmatched.csv)")
uploaded = files.upload()
file_used = [f for f in uploaded if 'PSmatched' in f][0]
df_all = pd.read_csv(file_used)

# === 2. Extract T/C groups and merge ===
T_cols = [col for col in df_all.columns if col.startswith('T_')]
C_cols = [col for col in df_all.columns if col.startswith('C_')]

df_T = df_all[T_cols].copy()
df_C = df_all[C_cols].copy()
df_T.columns = [col.replace('T_', '') for col in df_T.columns]
df_C.columns = [col.replace('C_', '') for col in df_C.columns]
df_T['Neoadjuvant hormone therapy'] = 1
df_C['Neoadjuvant hormone therapy'] = 0
df = pd.concat([df_T, df_C], ignore_index=True)

# === 3. Prepare and clean data ===
features = ['PSA POM1', 'PSA POM3', 'PSA POM6', 'PSA POM9', 'PSA POM12', 'Age', 'iPSA', 'Neoadjuvant hormone therapy']
targets = ['t_star', 'max_a', 'PSA POM36', 'Cluster']

for col in features + targets:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df[features + targets].dropna()
print(f"✅ Valid rows after cleaning: {len(df)}")

# === 4. Apply regression models (predefined coefficients) ===
tstar_coef = {'PSA POM1': 10.7578, 'PSA POM3': -11.4016, 'PSA POM6': 1.6251,
              'PSA POM9': 0.5301, 'PSA POM12': -1.7559, 'Age': 0.3539,
              'iPSA': -0.1319, 'Neoadjuvant hormone therapy': 0.1331}
tstar_intercept = -6.7723

maxa_coef = {'PSA POM1': 1.4707, 'PSA POM3': -1.0762, 'PSA POM6': 0.1744,
             'PSA POM9': 0.2299, 'PSA POM12': -0.7798, 'Age': 0.0044,
             'iPSA': -0.0006, 'Neoadjuvant hormone therapy': 0.0014}
maxa_intercept = -0.1342

psa36_coef = {'PSA POM1': 0.5901, 'PSA POM3': -1.7241, 'PSA POM6': -10.2079,
              'PSA POM9': 22.3819, 'PSA POM12': 8.7631, 'Age': 0.0862,
              'iPSA': -0.0105, 'Neoadjuvant hormone therapy': 0.0323}
psa36_intercept = -5.908

def apply_model(X, coef_dict, intercept):
    return X.dot(pd.Series(coef_dict)) + intercept

X = df[features].astype(float)
df['pred_t_star'] = apply_model(X, tstar_coef, tstar_intercept)
df['pred_max_a'] = apply_model(X, maxa_coef, maxa_intercept)
df['pred_psa36'] = apply_model(X, psa36_coef, psa36_intercept)

# === 5. Cluster classification ===
X_cluster = df[['pred_t_star', 'pred_max_a', 'pred_psa36']]
y_cluster = df['Cluster']
X_train, X_test, y_train, y_test = train_test_split(X_cluster, y_cluster, test_size=0.3, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n📊 Cluster classification based on D-module predictions:")
print(classification_report(y_test, y_pred))

# === 6. Baseline model: Age + iPSA + NHT → PSA@36M ===
X_conv = df[['Age', 'iPSA', 'Neoadjuvant hormone therapy']]
y_conv = df['PSA POM36']
model_conv = LinearRegression().fit(X_conv, y_conv)
y_conv_pred = model_conv.predict(X_conv)

print("\n📉 Baseline model for PSA@36M:")
print(f"R² = {r2_score(y_conv, y_conv_pred):.4f}")
print(f"MSE = {mean_squared_error(y_conv, y_conv_pred):.4f}")
