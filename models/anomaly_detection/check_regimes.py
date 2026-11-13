import pandas as pd

df = pd.read_parquet("data/processed/metropt3_labeled.parquet")
df["day"] = df["timestamp"].dt.floor("D")

# 每天:加载时间占比、TP2、H1、电机电流、故障标签占比
daily = df.groupby("day").agg(
    rows=("TP2", "size"),
    load=("DV_eletric", "mean"),
    TP2=("TP2", "mean"),
    H1=("H1", "mean"),
    current=("Motor_current", "mean"),
    failure=("failure", "mean"),
)
daily = daily[daily["rows"] > 1000]  # 排除数据太少的天

print("== 加载占比高、但不在故障表里的日子 ==")
sus = daily[(daily["load"] > 0.5) & (daily["failure"] < 0.5)]
print(sus.round(3).to_string() if len(sus) else "none")

print("\n== 每月平均 ==")
m = df.groupby(df["timestamp"].dt.to_period("M"))[["DV_eletric", "TP2", "H1", "Motor_current", "Oil_temperature"]].mean()
print(m.round(3))