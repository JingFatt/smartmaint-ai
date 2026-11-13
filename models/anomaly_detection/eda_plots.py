from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_parquet("data/processed/metropt3_labeled.parquet")
Path("docs/figures").mkdir(parents=True, exist_ok=True)

FAILURES = [
    ("2020-04-18 00:00", "2020-04-18 23:59"),
    ("2020-05-29 23:30", "2020-05-30 06:00"),
    ("2020-06-05 10:00", "2020-06-07 14:30"),
    ("2020-07-15 14:30", "2020-07-15 19:00"),
]
SIGNALS = ["TP2", "TP3", "Reservoirs", "Oil_temperature", "Motor_current"]


def plot(data, name, title):
    fig, axes = plt.subplots(len(SIGNALS), 1, figsize=(14, 10), sharex=True)
    for ax, s in zip(axes, SIGNALS):
        ax.plot(data.index, data[s], lw=0.7)
        ax.set_ylabel(s)
        for a, b in FAILURES:
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="red", alpha=0.25)
    axes[0].set_title(title)
    plt.tight_layout()
    plt.savefig(f"docs/figures/{name}.png", dpi=110)
    plt.close()


ts = df.set_index("timestamp")

# 1. 全程总览(10 分钟平均,断档处会自动断线)
plot(ts[SIGNALS].resample("10min").mean(), "overview", "Overview (10-min mean), red = reported failures")

# 2. 每次故障前 3 天到后 1 天的放大图(5 分钟平均)
for i, (a, b) in enumerate(FAILURES, start=1):
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    w = ts.loc[a - pd.Timedelta(days=3): b + pd.Timedelta(days=1), SIGNALS]
    plot(w.resample("5min").mean(), f"failure_{i}", f"Failure {i}: 3 days before to 1 day after")

# 3. 正常 vs 故障时段的各列平均值
cols = [c for c in df.columns if c not in ("timestamp", "failure", "failure_id")]
print(df.groupby("failure")[cols].mean().T.round(3))
print("figures saved to docs/figures")