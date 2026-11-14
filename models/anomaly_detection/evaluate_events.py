import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

FEATURES = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature",
            "Motor_current", "COMP", "DV_eletric", "Towers", "MPG", "LPS",
            "Pressure_switch", "Oil_level", "Caudal_impulses"]
ART = "models/anomaly_detection/artifacts/"
WIN = 180  # 180 行 x 10 秒 = 30 分钟

df = (pd.read_parquet("data/processed/metropt3_flagged.parquet")
      .sort_values("timestamp").reset_index(drop=True))

model = joblib.load(ART + "isolation_forest.joblib")
scaler = joblib.load(ART + "scaler.joblib")
with open(ART + "threshold.json") as f:
    t = json.load(f)
thr = t["threshold"] if isinstance(t, dict) and "threshold" in t else (
    next(iter(t.values())) if isinstance(t, dict) else float(t))

df["score"] = -model.score_samples(scaler.transform(df[FEATURES]))
df["flag_if"] = df["score"] > thr

# 简单规则:过去 30 分钟的满载比例
df["load30"] = df["DV_eletric"].rolling(WIN, min_periods=1).mean()
df["flag_rule"] = df["load30"] > 0.9

# 复现原来的划分
idx = df.index[df["clean_normal"]]
test_start = df.loc[idx[int(len(idx) * 0.85)], "timestamp"]
df["is_normal_test"] = df["clean_normal"] & (df["timestamp"] >= test_start)
df["is_fault_test"] = (df["failure"] == 1) & ~df["frozen"] & ~df["uncertain"]
test = df[df["is_normal_test"] | df["is_fault_test"]]
y = test["is_fault_test"].astype(int)

# ---- 1. 复现检查(应该约等于 FP 1127, FN 996)----
fp = int(((test["score"] > thr) & (y == 0)).sum())
fn = int(((test["score"] <= thr) & (y == 1)).sum())
print(f"sanity check -> FP={fp} (expect ~1127), FN={fn} (expect ~996)")

# ---- 2. 与简单规则比较 ----
print("\n== 行级指标(同一个测试集)==")
for name, s in [("IsolationForest", test["score"]), ("rule: load30", test["load30"])]:
    print(f"{name:16s} ROC-AUC={roc_auc_score(y, s):.4f}  PR-AUC={average_precision_score(y, s):.4f}")

# ---- 3. 各类时段被报警的比例 ----
month = df["timestamp"].dt.month
groups = {
    "normal_test": df["is_normal_test"],
    "failure 1": df["is_fault_test"] & (df["failure_id"] == 1),
    "failure 2": df["is_fault_test"] & (df["failure_id"] == 2),
    "failure 3": df["is_fault_test"] & (df["failure_id"] == 3),
    "failure 4": df["is_fault_test"] & (df["failure_id"] == 4),
    "uncertain (Mar)": df["uncertain"] & (month == 3),
    "uncertain (Apr+)": df["uncertain"] & (month >= 4),
    "frozen": df["frozen"],
}
rows = []
for k, m in groups.items():
    rows.append({"group": k, "rows": int(m.sum()),
                 "IF_flagged_%": round(100 * df.loc[m, "flag_if"].mean(), 1),
                 "rule_flagged_%": round(100 * df.loc[m, "flag_rule"].mean(), 1)})
print("\n== 各类时段被判为异常的比例 ==")
print(pd.DataFrame(rows).to_string(index=False))

# ---- 4. 事件级:报警延迟和误报 ----
df["alarm_if"] = df["flag_if"].rolling(WIN, min_periods=1).mean() > 0.5
df["alarm_rule"] = df["flag_rule"]

print("\n== 报警延迟(分钟,相对报告的故障开始;负数 = 提前报警)==")
for k in [1, 2, 3, 4]:
    ev = df[df["failure_id"] == k]
    start, end = ev["timestamp"].iloc[0], ev["timestamp"].iloc[-1]
    win = df[(df["timestamp"] >= start - pd.Timedelta(hours=2)) & (df["timestamp"] <= end)]
    out = {}
    for a in ["alarm_if", "alarm_rule"]:
        hit = win[win[a]]
        out[a] = round((hit["timestamp"].iloc[0] - start).total_seconds() / 60, 1) if len(hit) else None
    print(f"failure {k}: IF={out['alarm_if']}  rule={out['alarm_rule']}")

nt = df["is_normal_test"]
days = df.loc[nt, "timestamp"].dt.normalize().nunique()
print(f"\n== 正常测试期的误报(共 {days} 天)==")
for a in ["alarm_if", "alarm_rule"]:
    s = df[a] & nt
    edges = int((s & ~s.shift(fill_value=False)).sum())
    print(f"{a}: {edges} 次报警,约 {edges / days:.2f} 次/天")