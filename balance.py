import pandas as pd

# 1. Baca file bersih
df = pd.read_csv("dataset.csv")

# 2. Buat kolom label (jika belum ada)
if 'label' not in df.columns:
    df['label'] = df['clicks'].apply(lambda x: 1 if x > 0 else 0)

# 3. Pisahkan Yes dan No
yes_df = df[df['label'] == 1]
no_df = df[df['label'] == 0]

# 4. Set target rasio No:Yes (misal 49:1 untuk target CTR ~2%)
target_ratio = 10   # Ubah ke 99 untuk target CTR ~1%, dst
target_no = len(yes_df) * target_ratio

# 5. Sampling No
no_sampled = no_df.sample(n=target_no, random_state=42, replace=True if len(no_df) < target_no else False)

# 6. Gabung & acak data
balanced_df = pd.concat([yes_df, no_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

# 7. Simpan file balanced
balanced_df.to_csv("dataset.csv", index=False)
print(f"dataset.csv")
print(f"Jumlah Yes (label=1): {len(yes_df)}")
print(f"Jumlah No  (label=0): {len(no_sampled)}")
print(f"Total baris: {len(balanced_df)}")
print(f"Rasio Yes:No = 1:{int(len(no_sampled)/len(yes_df))}")
