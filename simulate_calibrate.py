from flask import Flask, request, jsonify
import pandas as pd
import pickle
import json

app = Flask(__name__)

# === Load model ===
with open("xgb_hris_ad_model.pkl", "rb") as f:
    model_data = pickle.load(f)

click_model = model_data["click_model"]
encoders = model_data.get("encoders", {})
features = model_data.get("features", [])
threshold = model_data.get("best_threshold", 0.13)

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    try:
        # Ambil JSON, dukung jika dikirim dari n8n stringified
        data = request.get_json(force=True)
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, list):
            return jsonify({"error": "Input harus berupa array JSON"}), 400

        df = pd.DataFrame(data)
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("__", "_")
        )

        # Simpan input asli
        original_df = df.copy()

        # Tambah kolom jika hilang
        for col in features:
            if col not in df.columns:
                df[col] = "Unknown" if col != "age" else 30

        df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(30).astype(int)

        # Encoding kategorikal
        for col, encoder in encoders.items():
            if col not in df.columns:
                continue
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
            df[col + "_enc"] = encoder.transform(df[col])

        # Susun X untuk prediksi
        X_parts = []
        for col in features:
            if col == "age":
                X_parts.append(df[[col]])
            else:
                X_parts.append(df[[col + "_enc"]])
        X = pd.concat(X_parts, axis=1)
        X.columns = features

        # Prediksi
        df["click_probability"] = click_model.predict_proba(X)[:, 1]
        df["respond"] = df["click_probability"].apply(lambda x: "Yes" if x > threshold else "No")

        # Gabungkan kembali dengan data asli
        result_df = original_df.copy()
        result_df["click_probability"] = df["click_probability"]
        result_df["respond"] = df["respond"]

        return jsonify(result_df.where(pd.notnull(result_df), None).to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5500)
