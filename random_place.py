import pandas as pd
import random

# Load CSV
input_file = "balanced_training_dataset.csv"  # Ganti jika file berbeda
df = pd.read_csv(input_file)

# Daftar nilai ad placement yang mungkin
placements = ["Instagram", "TikTok", "LinkedIn"]

# Acak isi kolom 'ad_placement'
df['ad_placement'] = [random.choice(placements) for _ in range(len(df))]

# Simpan ke file baru
output_file = "data_randomized.csv"
df.to_csv(output_file, index=False)

print(f"✅ Kolom 'ad_placement' telah diacak dan disimpan ke '{output_file}'")
