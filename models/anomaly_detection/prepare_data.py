import pandas as pd

RAW = "data/raw/MetroPT3(AirCompressor).csv"
OUT = "data/processed/metropt3_labeled.parquet"

# 来自 Data Description_Metro.pdf 的故障报告
FAILURES = [
    ("2020-04-18 00:00", "2020-04-18 23:59"),
    ("2020-05-29 23:30", "2020-05-30 06:00"),
    ("2020-06-05 10:00", "2020-06-07 14:30"),
    ("2020-07-15 14:30", "2020-07-15 19:00"),
]


def main():
    df = pd.read_csv(RAW, parse_dates=["timestamp"]).drop(columns=["Unnamed: 0"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["failure"] = 0
    df["failure_id"] = 0
    for i, (start, end) in enumerate(FAILURES, start=1):
        mask = df["timestamp"].between(pd.Timestamp(start), pd.Timestamp(end))
        df.loc[mask, "failure"] = 1
        df.loc[mask, "failure_id"] = i
        print(f"failure {i}: {start} -> {end}, rows = {mask.sum()}")
    print("failure rows:", df["failure"].sum(), f"({df['failure'].mean():.2%})")

    # 找出超过 10 分钟的数据断档
    gap = df["timestamp"].diff().dt.total_seconds()
    big = gap[gap > 600].sort_values(ascending=False)
    print("gaps > 10 min:", len(big))
    for idx, g in big.head(10).items():
        print(df.loc[idx - 1, "timestamp"], "->", df.loc[idx, "timestamp"], f"{g / 3600:.1f} h")

    df.to_parquet(OUT, index=False)
    print("saved:", OUT, df.shape)


if __name__ == "__main__":
    main()