import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature",
            "Motor_current", "COMP", "DV_eletric", "Towers", "MPG", "LPS",
            "Pressure_switch", "Oil_level", "Caudal_impulses"]
MODEL_NAME = "smartmaint-anomaly-detector"
EXPERIMENT = "smartmaint-anomaly"
PARAMS = dict(n_estimators=300, max_samples=4096, contamination="auto",
              random_state=7, n_jobs=-1)

# ---------- 训练:只用干净的正常数据 ----------
train = pd.read_parquet("data/processed/train_normal.parquet")[FEATURES].astype("float64").to_numpy()
val = pd.read_parquet("data/processed/validation.parquet")[FEATURES].astype("float64").to_numpy()

scaler = StandardScaler().fit(train)
model = IsolationForest(**PARAMS).fit(scaler.transform(train))
pipe = Pipeline([("scaler", scaler), ("iforest", model)])
thr = float(np.percentile(-pipe.score_samples(val), 99))   # 与 champion 相同的定阈值方法
print("threshold:", thr)

# ---------- 测试集:与 log_models.py 完全相同 ----------
df = (pd.read_parquet("data/processed/metropt3_flagged.parquet")
      .sort_values("timestamp").reset_index(drop=True))
idx = df.index[df["clean_normal"]]
test_start = df.loc[idx[int(len(idx) * 0.85)], "timestamp"]
df["is_normal_test"] = df["clean_normal"] & (df["timestamp"] >= test_start)
df["is_fault_test"] = (df["failure"] == 1) & ~df["frozen"] & ~df["uncertain"]
test = df[df["is_normal_test"] | df["is_fault_test"]].copy()

X = test[FEATURES].astype("float64")
test["score"] = -pipe.score_samples(X.to_numpy())
test["flag"] = test["score"] > thr

y = test["is_fault_test"].astype(int)
flags = test["flag"].astype(bool)
tp = int((flags & (y == 1)).sum())
fp = int((flags & (y == 0)).sum())
fn = int((~flags & (y == 1)).sum())
metrics = {
    "roc_auc": roc_auc_score(y, test["score"]),
    "pr_auc": average_precision_score(y, test["score"]),
    "precision": tp / (tp + fp),
    "recall": tp / (tp + fn),
    "false_positives": fp,
    "false_negatives": fn,
}
for k in [1, 2, 3, 4]:
    sel = test["is_fault_test"] & (test["failure_id"] == k)
    metrics[f"flagged_pct_failure_{k}"] = 100 * float(test.loc[sel, "flag"].mean())

# ---------- 记录并注册新版本(不改 champion 别名)----------
mlflow.set_tracking_uri("sqlite:///mlops/mlflow.db")
mlflow.set_experiment(EXPERIMENT)
with mlflow.start_run(run_name="isolation_forest_v2_challenger"):
    mlflow.set_tags({"role": "challenger", "task": "compressor anomaly detection"})
    mlflow.log_params({"model_type": "IsolationForest+StandardScaler", "threshold": thr,
                       "data": "metropt3_flagged.parquet",
                       **{f"if_{k}": v for k, v in model.get_params().items()}})
    mlflow.log_metrics(metrics)
    mlflow.log_dict({"threshold": thr, "features": FEATURES}, "threshold.json")
    info = mlflow.sklearn.log_model(
        pipe, name="model", input_example=X.head(5),
        registered_model_name=MODEL_NAME,
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )
    print("registered challenger as version", info.registered_model_version)
print({k: round(v, 4) for k, v in metrics.items()})