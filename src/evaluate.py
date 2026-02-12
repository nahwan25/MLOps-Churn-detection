import os, json
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Load test data
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# Load model
model = load("model/model.joblib")

# Prediksi
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

# Hitung metrics
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred)
}

if y_proba is not None:
    metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

# Simpan metrics di folder metrics/
os.makedirs("metrics", exist_ok=True)
output_path = os.path.join("metrics", "metrics.json")
with open(output_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Evaluasi selesai. Metrics disimpan di {output_path}")
