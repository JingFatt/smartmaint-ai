import pandas as pd

df = pd.read_parquet("data/processed/metropt3_labeled.parquet")
df = df.sort_values("timestamp").reset_index(drop=True)

loaded = df["DV_eletric"] > 0.5
gap = df["timestamp"].diff().dt.total_seconds().fillna(0) > 60  # 数据断档
run_id = ((loaded != loaded.shift()) | gap).cumsum()

rows = []
for rid, g in df[loaded].groupby(run_id[loaded]):
    hours = (g["timestamp"].iloc[-1] - g["timestamp"].iloc[0]).total_seconds() / 3600
    if hours >= 1:
        rows.append({
            "start": g["timestamp"].iloc[0],
            "end": g["timestamp"].iloc[-1],
            "hours": round(hours, 1),
            "TP2_mean": round(g["TP2"].mean(), 3),
            "TP2_std": round(g["TP2"].std(), 4),
            "cur_mean": round(g["Motor_current"].mean(), 3),
            "cur_std": round(g["Motor_current"].std(), 4),
            "labeled": round(g["failure"].mean(), 2),
        })

out = pd.DataFrame(rows)
print(f"sustained-load episodes >= 1h: {len(out)}")
print(out.to_string(index=False))