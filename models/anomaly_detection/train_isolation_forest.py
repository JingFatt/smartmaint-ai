
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# ==========================================
# 1. Configuration
# ==========================================

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/anomaly_detection/artifacts")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "TP2", "TP3", "H1", "DV_pressure",
    "Reservoirs", "Oil_temperature", "Motor_current",
    "COMP", "DV_eletric", "Towers", "MPG",
    "LPS", "Pressure_switch", "Oil_level",
    "Caudal_impulses"
]

RANDOM_STATE = 42
N_ESTIMATORS = 300
MAX_SAMPLES = 256

# ==========================================
# 2. Load datasets
# ==========================================

print("\nLoading datasets...")

train = pd.read_parquet(DATA_DIR / "train_normal.parquet")
validation = pd.read_parquet(DATA_DIR / "validation.parquet")
test = pd.read_parquet(DATA_DIR / "test.parquet")

X_train = train[FEATURES].to_numpy(dtype=np.float32)
X_val = validation[FEATURES].to_numpy(dtype=np.float32)
X_test = test[FEATURES].to_numpy(dtype=np.float32)

y_test = test["target"].to_numpy(dtype=int)

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# ==========================================
# 3. Feature Scaling
# ==========================================

print("\nFitting StandardScaler...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ==========================================
# 4. Train Isolation Forest
# ==========================================

print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    max_samples=MAX_SAMPLES,
    contamination="auto",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train)

print("Training completed.")

# ==========================================
# 5. Calibrate threshold
# ==========================================

print("\nCalibrating anomaly threshold...")

# Lower score_samples = more anomalous.
# Negate it so higher scores mean more anomalous.

val_scores = -model.score_samples(X_val)

threshold = float(np.percentile(val_scores, 99))

print("Anomaly threshold:", threshold)

# ==========================================
# 6. Evaluate test dataset
# ==========================================

print("\nEvaluating test dataset...")

test_scores = -model.score_samples(X_test)

y_pred = (test_scores >= threshold).astype(int)

print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, y_pred))

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Anomaly"],
        zero_division=0
    )
)

print("\n========== ADDITIONAL METRICS ==========")

print("ROC-AUC:", roc_auc_score(y_test, test_scores))

print(
    "PR-AUC (Average Precision):",
    average_precision_score(y_test, test_scores)
)

# ==========================================
# 7. Save model and scaler
# ==========================================

joblib.dump(
    model,
    MODEL_DIR / "isolation_forest.joblib"
)

joblib.dump(
    scaler,
    MODEL_DIR / "scaler.joblib"
)

with open(MODEL_DIR / "threshold.json", "w") as f:
    json.dump(
        {
            "threshold": threshold,
            "percentile": 99,
            "features": FEATURES,
            "model": "IsolationForest",
            "n_estimators": N_ESTIMATORS,
            "max_samples": MAX_SAMPLES,
            "random_state": RANDOM_STATE
        },
        f,
        indent=4
    )

print("\n========== SAVED ARTIFACTS ==========")
print(MODEL_DIR / "isolation_forest.joblib")
print(MODEL_DIR / "scaler.joblib")
print(MODEL_DIR / "threshold.json")

print("\nModel training pipeline completed.")
