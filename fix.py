import pandas as pd
import pickle

# === CONFIG ===
input_csv = "Simulasi_Prediksi_20250725_170742 - sheet1.csv"
output_csv = "Simulasi_Prediksi_fixed_output.csv"
model_file = "xgb_hris_ad_model.pkl"

# === Load model ===
with open(model_file, "rb") as f:
    model_data = pickle.load(f)

click_model = model_data["click_model"]
encoders = model_data.get("encoders", {})
features = model_data.get("features", [])
threshold = model_data.get("threshold", 0.13)

# === Load CSV dan normalisasi kolom ===
df = pd.read_csv(input_csv, sep=";")
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("__", "_")
)

print("📄 Kolom setelah normalisasi:")
print(df.columns.tolist())

# === Tambahkan kolom dummy jika hilang
for col in features:
    if col not in df.columns:
        print(f"⚠️  Kolom '{col}' tidak ditemukan. Menambahkan kolom dummy.")
        df[col] = 30 if col == "age" else "Unknown"

# === Lakukan encoding
for col, encoder in encoders.items():
    if col not in df.columns:
        print(f"⚠️  Kolom '{col}' tidak tersedia di data, lewati.")
        continue

    df[col] = df[col].astype(str).str.strip()
    known_classes = set(encoder.classes_)
    df[col] = df[col].apply(lambda x: x if x in known_classes else list(known_classes)[0])
    df[col + "_enc"] = encoder.transform(df[col])

# === Buat X untuk prediksi
X_parts = []
for col in features:
    if col == "age":
        X_parts.append(df[[col]])
    else:
        X_parts.append(df[[col + "_enc"]])

X = pd.concat(X_parts, axis=1)
X.columns = features

# === Prediksi
click_proba = click_model.predict_proba(X)[:, 1]
df["click_probability"] = click_proba
df["respond"] = df["click_probability"].apply(lambda x: "Yes" if x > threshold else "No")

# === Log Threshold dan Contoh
print(f"\n✅ Threshold digunakan: {threshold:.2f}")
print("\n🎯 Contoh hasil:")
for idx, row in df.head(5).iterrows():
    print(f"\nAudience {idx + 1}")
    print(f"Gender          : {row.get('gender')}")
    print(f"Age             : {row.get('age')}")
    print(f"Ad type         : {row.get('ad_type')}")
    print(f"Ad topic        : {row.get('ad_topic')}")
    print(f"Ad placement    : {row.get('ad_placement')}")
    print(f"Click Probability : {row['click_probability']:.4f}")
    print(f"Respond           : {row['respond']}")

# === Hitung jumlah Yes/No dan persentase
yes_count = (df["respond"] == "Yes").sum()
no_count = (df["respond"] == "No").sum()
total = len(df)

yes_pct = yes_count / total * 100
no_pct = no_count / total * 100

print(f"\n📊 Ringkasan Respond:")
print(f"Yes: {yes_count} ({yes_pct:.2f}%)")
print(f"No : {no_count} ({no_pct:.2f}%)")

# === Simpan hasil
df.to_csv(output_csv, sep=";", index=False)
print(f"\n💾 File disimpan sebagai: {output_csv}")
