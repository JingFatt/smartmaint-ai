import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

FEATURES = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature",
            "Motor_current", "COMP", "DV_eletric", "Towers", "MPG", "LPS",
            "Pressure_switch", "Oil_level", "Caudal_impulses"]
ART = Path("models/anomaly_detection/artifacts")
TRACKING_URI = "sqlite:///mlops/mlflow.db"
ARTIFACT_ROOT = Path("mlops/mlartifacts").resolve()
EXPERIMENT = "smartmaint-anomaly"
MODEL_NAME = "smartmaint-anomaly-detector"
WIN = 180          # 180 行 x 10 秒 = 30 分钟
RULE_THR = 0.9

# ---------- 载入模型与数据 ----------
model = joblib.load(ART / "isolation_forest.joblib")
scaler = joblib.load(ART / "scaler.joblib")
with open(ART / "threshold.json") as f:
    t = json.load(f)
thr = t["threshold"] if isinstance(t, dict) and "threshold" in t else (
    next(iter(t.values())) if isinstance(t, dict) else float(t))

df = (pd.read_parquet("data/processed/metropt3_flagged.parquet")
      .sort_values("timestamp").reset_index(drop=True))
df["load30"] = df["DV_eletric"].rolling(WIN, min_periods=1).mean()
df["flag_rule"] = df["load30"] > RULE_THR

# 复现和 evaluate_events.py 相同的测试集
idx = df.index[df["clean_normal"]]
test_start = df.loc[idx[int(len(idx) * 0.85)], "timestamp"]
df["is_normal_test"] = df["clean_normal"] & (df["timestamp"] >= test_start)
df["is_fault_test"] = (df["failure"] == 1) & ~df["frozen"] & ~df["uncertain"]
test = df[df["is_normal_test"] | df["is_fault_test"]].copy()

X = test[FEATURES].astype("float64")
test["score_if"] = -model.score_samples(scaler.transform(X))
test["flag_if"] = test["score_if"] > thr

# Pipeline 必须和原来的 scaler + model 给出完全相同的分数
pipe = Pipeline([("scaler", scaler), ("iforest", model)])
assert np.allclose(-pipe.score_samples(X.head(200)), test["score_if"].head(200)), "pipeline mismatch"


def summarize(frame, score_col, flag_col):
    y = frame["is_fault_test"].astype(int)
    flags = frame[flag_col].astype(bool)
    tp = int((flags & (y == 1)).sum())
    fp = int((flags & (y == 0)).sum())
    fn = int((~flags & (y == 1)).sum())
    m = {
        "roc_auc": roc_auc_score(y, frame[score_col]),
        "pr_auc": average_precision_score(y, frame[score_col]),
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "false_positives": fp,
        "false_negatives": fn,
    }
    for k in [1, 2, 3, 4]:
        sel = frame["is_fault_test"] & (frame["failure_id"] == k)
        m[f"flagged_pct_failure_{k}"] = 100 * float(frame.loc[sel, flag_col].mean())
    return m


# ---------- MLflow ----------
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()
if client.get_experiment_by_name(EXPERIMENT) is None:
    mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_ROOT.as_uri())
mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name="rule_load30"):
    mlflow.set_tags({"role": "baseline", "task": "compressor anomaly detection"})
    mlflow.log_params({"model_type": "rule", "window_rows": WIN,
                       "load_threshold": RULE_THR, "data": "metropt3_flagged.parquet"})
    mlflow.log_metrics(summarize(test, "load30", "flag_rule"))
    print("logged: rule_load30")

with mlflow.start_run(run_name="isolation_forest_v1"):
    mlflow.set_tags({"role": "candidate", "task": "compressor anomaly detection",
                     "limitation": "only 4 reported failures; failure 4 differs from 1-3"})
    mlflow.log_params({"model_type": "IsolationForest+StandardScaler",
                       "threshold": thr, "data": "metropt3_flagged.parquet",
                       **{f"if_{k}": v for k, v in model.get_params().items()}})
    mlflow.log_metrics(summarize(test, "score_if", "flag_if"))
    mlflow.log_dict({"threshold": thr, "features": FEATURES}, "threshold.json")
    if Path("docs/eval_v1.txt").exists():
        mlflow.log_artifact("docs/eval_v1.txt")
    info = mlflow.sklearn.log_model(
        pipe, name="model",
        input_example=X.head(5),
        registered_model_name=MODEL_NAME,
	skops_trusted_types=["sklearn.tree._tree.Tree"],
    )
    version = info.registered_model_version
    client.set_registered_model_alias(MODEL_NAME, "champion", version)
    print(f"logged: isolation_forest_v1 -> {MODEL_NAME} version {version}, alias champion")