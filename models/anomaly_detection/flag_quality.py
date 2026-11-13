import pandas as pd

df = pd.read_parquet("data/processed/metropt3_labeled.parquet")
df = df.sort_values("timestamp").reset_index(drop=True)
pad = pd.Timedelta(hours=1)

# 1. 卡死:四个模拟信号连续 360 行以上完全不变
sig = ["TP2", "TP3", "Motor_current", "Oil_temperature"]
same = df[sig].diff().abs().sum(axis=1) == 0
grp = (~same).cumsum()
df["frozen"] = grp.map(grp.value_counts()) >= 360

# 2. 持续加载时段:有报告的 -> failure_zone;没报告的 -> uncertain
loaded = df["DV_eletric"] > 0.5
gap = df["timestamp"].diff().dt.total_seconds().fillna(0) > 60
run_id = ((loaded != loaded.shift()) | gap).cumsum()

df["failure_zone"] = False
df["uncertain"] = False
for rid, g in df[loaded].groupby(run_id[loaded]):
    start, end = g["timestamp"].iloc[0], g["timestamp"].iloc[-1]
    if end - start < pd.Timedelta(hours=1):
        continue
    col = "failure_zone" if g["failure"].mean() > 0.5 else "uncertain"
    df.loc[df["timestamp"].between(start - pad, end + pad), col] = True

# 3. 可以安全当作"正常"的数据
df["clean_normal"] = ~(df["frozen"] | df["failure_zone"] | df["uncertain"]) & (df["failure"] == 0)

df.to_parquet("data/processed/metropt3_flagged.parquet", index=False)

df["month"] = df["timestamp"].dt.to_period("M")
summary = df.groupby("month").agg(
    rows=("failure", "size"),
    clean_normal=("clean_normal", "sum"),
    frozen=("frozen", "sum"),
    uncertain=("uncertain", "sum"),
    failure=("failure", "sum"),
)
print(summary)
print()
print(df[["frozen", "uncertain", "failure_zone", "clean_normal"]].sum())