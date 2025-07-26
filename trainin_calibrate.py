import pandas as pd
import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

# === Load Data CSV ===
csv_file = "dataset.csv"  # Ganti dengan nama file kamu
df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip().str.lower()

# === Deteksi kolom target (tambahkan 'label') ===
possible_targets = ['clicked', 'click', 'respond', 'label']
target_col = next((col for col in df.columns if col in possible_targets), None)
if not target_col:
    raise ValueError("❌ Tidak ditemukan kolom target seperti 'clicked', 'click', 'respond', atau 'label'.")

df[target_col] = df[target_col].astype(int)
df['age'] = df['age'].fillna(30)

# === Isi default kolom iklan jika tidak tersedia
for col in ['gender', 'ad_type', 'ad_topic', 'ad_placement']:
    if col not in df.columns:
        df[col] = 'Unknown'

# === Encode kolom kategori ===
encoders = {}
for col in ['gender', 'ad_type', 'ad_topic', 'ad_placement']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# === Fitur yang dipakai untuk training ===
features = ['gender', 'age', 'ad_type', 'ad_topic', 'ad_placement']

X = df[features]
y = df[target_col]

# === Split data ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Model konfigurasi
click_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    reg_alpha=0.2,
    reg_lambda=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# === Training
click_model.fit(X_train, y_train)

# === Cari threshold terbaik (F1)
val_proba = click_model.predict_proba(X_val)[:, 1]
best_f1 = 0
best_threshold = 0.5

for t in np.arange(0.01, 0.99, 0.01):
    pred = (val_proba >= t).astype(int)
    f1 = f1_score(y_val, pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"\n✅ Best Threshold (F1-optimized): {round(best_threshold, 3)}")

# === Simpan model
model_data = {
    'click_model': click_model,
    'encoders': encoders,
    'features': features,
    'best_threshold': best_threshold
}
with open("xgb_hris_ad_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("📦 Model disimpan ke xgb_hris_ad_model.pkl")
