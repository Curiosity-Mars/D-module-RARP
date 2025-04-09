# dmodule_psa_dynamics_prediction_pipeline.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, classification_report
from google.colab import files
from io import StringIO

# === 1. Upload CSV File ===
print("⬆️ Upload CSV file with T_XXX / C_XXX columns")
uploaded = files.upload()
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)

# === 2. Display uploaded columns and preview ===
print("\n✅ Uploaded columns:")
print(df.columns.tolist())
print("\n✅ Preview of 1st row:")
print(df.head(1))

input("\n▶️ Press Enter to continue if the structure looks correct...")

# === 3. Define columns ===
psa_cols = ['PSA POM1', 'PSA POM3', 'PSA POM6', 'PSA POM9', 'PSA POM12']
clinical_cols = ['Age', 'iPSA', 'Neoadjuvant hormone therapy']
target_cols = ['t_star', 'max_a', 'PSA POM36']

t_dict = {f'T_{col}': col for col in psa_cols + clinical_cols + target_cols}
c_dict = {f'C_{col}': col for col in psa_cols + clinical_cols + target_cols}

df_t = df[[col for col in df.columns if col in t_dict]].rename(columns=t_dict, errors='ignore')
df_c = df[[col for col in df.columns if col in c_dict]].rename(columns=c_dict, errors='ignore')

df_all = pd.concat([df_t, df_c], ignore_index=True)
print(f"\n📊 Merged data shape (before dropna): {df_all.shape}")

df_all = df_all.apply(pd.to_numeric, errors='coerce').dropna()
print(f"📉 After dropna: {df_all.shape}")

if df_all.empty:
    raise ValueError("❌ No valid rows left after dropna. Check missing values in PSA/t_star.")

# === 4. Prepare Features and Targets ===
X = df_all[psa_cols + clinical_cols]
y_tstar = df_all['t_star']
y_maxa = df_all['max_a']
y_psa36 = df_all['PSA POM36']

# === 5. Define Regression Function ===
def train_model(X, y):
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    return {
        'intercept': model.intercept_,
        'coefficients': dict(zip(X.columns, model.coef_)),
        'R2': r2_score(y, y_pred),
        'MSE': mean_squared_error(y, y_pred)
    }

# === 6. Train and Report ===
print("\n📈 [Prediction Results]")

result_tstar = train_model(X, y_tstar)
result_maxa = train_model(X, y_maxa)
result_psa36 = train_model(X, y_psa36)

print("\n🔹 Predicting t_star:\n", result_tstar)
print("\n🔹 Predicting max_a:\n", result_maxa)
print("\n🔹 Predicting PSA@36M:\n", result_psa36)

# === 7. Optional Cluster Classification ===
if 'T_Cluster' in df.columns:
    print("\n🧪 [Cluster Classification with Random Forest]")
    df_all['Cluster'] = pd.to_numeric(df['T_Cluster'], errors='coerce')
    df_all.dropna(subset=['Cluster'], inplace=True)

    cluster_X = df_all[['t_star', 'max_a', 'PSA POM36']]
    cluster_y = df_all['Cluster']

    clf = RandomForestClassifier(random_state=42)
    clf.fit(cluster_X, cluster_y)
    pred_cluster = clf.predict(cluster_X)

    print("\n📋 Classification Report:")
    print(classification_report(cluster_y, pred_cluster))
