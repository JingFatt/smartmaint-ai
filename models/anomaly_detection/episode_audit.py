from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_parquet("data/processed/metropt3_labeled.parquet")
df = df.sort_values("timestamp").reset_index(drop=True)
Path("docs/figures").mkdir(parents=True, exist_ok=True)

# ---------- 1. 持续加载时段,加上更多信号 ----------
loaded = df["DV_eletric"] > 0.5
gap = df["timestamp"].diff().dt.total_seconds().fillna(0) > 60
run_id = ((loaded != loaded.shift()) | gap).cumsum()

eps = []
for rid, g in df[loaded].groupby(run_id[loaded]):
    hours = (g["timestamp"].iloc[-1] - g["timestamp"].iloc[0]).total_seconds() / 3600
    if hours >= 1:
        eps.append({
            "start": g["timestamp"].iloc[0], "end": g["timestamp"].iloc[-1],
            "hours": round(hours, 1),
            "TP2": g["TP2"].mean(), "TP3": g["TP3"].mean(), "TP3_std": g["TP3"].std(),
            "H1": g["H1"].mean(), "DVp": g["DV_pressure"].mean(),
            "oilT": g["Oil_temperature"].mean(), "labeled": g["failure"].mean(),
        })
ep = pd.DataFrame(eps).round(3)
ep["label"] = np.where(ep["labeled"] > 0.5, "REPORTED", "unreported")
ep.to_csv("docs/episodes.csv", index=False)
print(ep.to_string())

# ---------- 2. 每个时段的放大图(前 12 小时到后 6 小时)----------
n = len(ep)
cols = 3
rows = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows, cols, figsize=(16, 3 * rows))
ts = df.set_index("timestamp")
for ax, (i, r) in zip(axes.ravel(), ep.iterrows()):
    w = ts.loc[r["start"] - pd.Timedelta(hours=12): r["end"] + pd.Timedelta(hours=6)]
    w = w[["TP2", "Oil_temperature"]].resample("5min").mean()
    ax.plot(w.index, w["TP2"], lw=0.6)
    ax2 = ax.twinx()
    ax2.plot(w.index, w["Oil_temperature"], lw=0.6, color="orange")
    ax.axvspan(r["start"], r["end"], color="red" if r["label"] == "REPORTED" else "gray", alpha=0.2)
    ax.set_title(f'#{i} {r["start"]:%m-%d %H:%M} {r["hours"]}h {r["label"]}', fontsize=9)
    ax.tick_params(labelsize=7)
    ax2.tick_params(labelsize=7)
for ax in axes.ravel()[n:]:
    ax.axis("off")
plt.tight_layout()
plt.savefig("docs/figures/episodes.png", dpi=100)
plt.close()

# ---------- 3. 卡死检查:4 个模拟信号连续多行完全不变 ----------
sig = ["TP2", "TP3", "Motor_current", "Oil_temperature"]
same = df[sig].diff().abs().sum(axis=1) == 0
grp = (~same).cumsum()
runs = df.groupby(grp).agg(start=("timestamp", "first"), end=("timestamp", "last"),
                           n=("TP2", "size"), failure=("failure", "mean"))
frozen = runs[runs["n"] >= 360].copy()  # 至少 360 行,约 1 小时
frozen["hours_of_rows"] = (frozen["n"] * 10 / 3600).round(1)
print(f"\nfrozen stretches: {len(frozen)}")
print(frozen[["start", "end", "n", "hours_of_rows", "failure"]].to_string(index=False))
print("figure saved: docs/figures/episodes.png")