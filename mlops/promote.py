import csv
from datetime import datetime
from pathlib import Path

import mlflow
from mlflow import MlflowClient

MODEL_NAME = "smartmaint-anomaly-detector"
MAX_EVENT_DROP = 2.0          # 任何一次故障的检出率最多下降多少个百分点
MAX_FP_RATIO = 1.10           # 误报行数最多是 champion 的多少倍
MIN_PR_AUC_GAIN = 0.005       # PR-AUC 最少提高多少
MIN_WORST_EVENT_GAIN = 5.0    # 或者最差故障的检出率最少提高多少个百分点

mlflow.set_tracking_uri("sqlite:///mlops/mlflow.db")
client = MlflowClient()

champ = client.get_model_version_by_alias(MODEL_NAME, "champion")
cand = max(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda v: int(v.version))
if int(cand.version) == int(champ.version):
    raise SystemExit("没有挑战者:最新版本就是 champion")

mc = client.get_run(champ.run_id).data.metrics
mn = client.get_run(cand.run_id).data.metrics
events = [f"flagged_pct_failure_{k}" for k in (1, 2, 3, 4)]

print(f"champion = v{champ.version}   challenger = v{cand.version}\n")
print(f"{'metric':26s}{'champion':>12s}{'challenger':>12s}")
for key in ["roc_auc", "pr_auc", "precision", "recall", "false_positives",
            "false_negatives"] + events:
    print(f"{key:26s}{mc[key]:12.4f}{mn[key]:12.4f}")

worst_drop = max(mc[e] - mn[e] for e in events)
gain_pr = mn["pr_auc"] - mc["pr_auc"]
gain_worst = min(mn[e] for e in events) - min(mc[e] for e in events)

checks = {
    f"no event drops by more than {MAX_EVENT_DROP} points (worst change: {-worst_drop:+.1f})":
        worst_drop <= MAX_EVENT_DROP,
    f"false positives <= {MAX_FP_RATIO}x champion ({mn['false_positives']:.0f} vs {mc['false_positives']:.0f})":
        mn["false_positives"] <= mc["false_positives"] * MAX_FP_RATIO,
    f"PR-AUC gain >= {MIN_PR_AUC_GAIN} (got {gain_pr:+.4f}) OR worst-event gain >= {MIN_WORST_EVENT_GAIN} (got {gain_worst:+.1f})":
        gain_pr >= MIN_PR_AUC_GAIN or gain_worst >= MIN_WORST_EVENT_GAIN,
}
print("\n== gate ==")
for text, ok in checks.items():
    print(("PASS  " if ok else "FAIL  ") + text)

promote = all(checks.values())
if promote:
    client.set_registered_model_alias(MODEL_NAME, "previous", champ.version)
    client.set_registered_model_alias(MODEL_NAME, "champion", cand.version)
    print(f"\nPROMOTED: champion is now v{cand.version} (previous = v{champ.version})")
else:
    client.set_model_version_tag(MODEL_NAME, cand.version, "gate", "rejected")
    print(f"\nREJECTED: champion stays v{champ.version}")

log = Path("docs/promotion_log.csv")
new = not log.exists()
with open(log, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["time", "champion", "challenger", "decision", "failed_checks"])
    w.writerow([datetime.now().isoformat(timespec="seconds"), champ.version, cand.version,
                "promoted" if promote else "rejected",
                "; ".join(t for t, ok in checks.items() if not ok)])
