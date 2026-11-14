import json
import shutil
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import skops.io as sio
from mlflow import MlflowClient

MODEL_NAME = "smartmaint-anomaly-detector"
OUT = Path("serving/model")
TMP = Path("serving/_download")
TRUSTED = ["sklearn.tree._tree.Tree"]   # 和注册时信任的类型一致
FEATURES = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature",
            "Motor_current", "COMP", "DV_eletric", "Towers", "MPG", "LPS",
            "Pressure_switch", "Oil_level", "Caudal_impulses"]

mlflow.set_tracking_uri("sqlite:///mlops/mlflow.db")
client = MlflowClient()

# 1. 找到 champion(这一步同时验证别名是否存在)
mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
print(f"champion = version {mv.version}, run_id = {mv.run_id}")

# 2. 下载模型文件
OUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)
local = Path(mlflow.artifacts.download_artifacts(
    artifact_uri=f"models:/{MODEL_NAME}@champion", dst_path=str(TMP)))
files = [p for p in local.rglob("*") if p.is_file()]
print("downloaded files:")
for p in files:
    print("  ", p.relative_to(local))

skops_files = [p for p in files if p.suffix == ".skops"]
if not skops_files:
    raise SystemExit("没有找到 .skops 文件,请把上面的文件列表贴给我")

# 3. 读取模型和阈值
pipe = sio.load(skops_files[0], trusted=TRUSTED)
run = client.get_run(mv.run_id)
thr = float(run.data.params["threshold"])

# 4. 核对:导出的模型必须和原来的 joblib 模型给出相同的分数
X = pd.read_parquet("data/processed/metropt3_flagged.parquet",
                    columns=FEATURES).head(500).astype("float64").to_numpy()
old_model = joblib.load("models/anomaly_detection/artifacts/isolation_forest.joblib")
old_scaler = joblib.load("models/anomaly_detection/artifacts/scaler.joblib")
old = -old_model.score_samples(old_scaler.transform(X))
new = -pipe.score_samples(X)
assert np.allclose(old, new), "导出的模型和原模型分数不一致"
print("score check: OK (500 rows identical)")

# 5. 保存给 API 使用的文件
joblib.dump(pipe, OUT / "model.joblib")
info = {
    "model_name": MODEL_NAME,
    "version": mv.version,
    "alias": "champion",
    "run_id": mv.run_id,
    "threshold": thr,
    "features": FEATURES,
    "window_rows": 180,
    "rule_load_threshold": 0.9,
}
(OUT / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
shutil.rmtree(TMP, ignore_errors=True)
print("exported to", OUT)
print(json.dumps(info, indent=2))