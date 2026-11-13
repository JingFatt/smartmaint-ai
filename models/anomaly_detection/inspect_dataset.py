
import pandas as pd

df = pd.read_parquet(
    "data/processed/metropt3_flagged.parquet"
)

df = df.sort_values("timestamp").reset_index(drop=True)

print("\n========== 1. TIMELINE ==========")

print("Start:", df["timestamp"].min())
print("End:", df["timestamp"].max())
print("Duration:", df["timestamp"].max() - df["timestamp"].min())

print("\n========== 2. FAILURE ID ==========")

print(df["failure_id"].value_counts().sort_index())

print("\n========== 3. ORIGINAL FAILURE EVENTS ==========")

events = (
    df[df["failure"] == 1]
    .groupby("failure_id")
    .agg(
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        rows=("failure", "size")
    )
)

print(events.to_string())

print("\n========== 4. FAILURE ZONE PERIODS ==========")

zone = df[df["failure_zone"]].copy()

zone["event_group"] = (
    zone["timestamp"].diff().gt(pd.Timedelta(minutes=1))
).cumsum()

periods = zone.groupby("event_group").agg(
    start=("timestamp", "min"),
    end=("timestamp", "max"),
    rows=("failure_zone", "size"),
    original_failure=("failure", "sum")
)

print(periods.to_string())

print("\n========== 5. MONTHLY LABEL DISTRIBUTION ==========")

df["month"] = df["timestamp"].dt.to_period("M")

monthly = df.groupby("month").agg(
    total=("timestamp", "size"),
    normal=("clean_normal", "sum"),
    frozen=("frozen", "sum"),
    uncertain=("uncertain", "sum"),
    failure_zone=("failure_zone", "sum"),
    original_failure=("failure", "sum")
)

print(monthly.to_string())

print("\n========== 6. SENSOR FEATURES ==========")

features = [
    c for c in df.select_dtypes(include="number").columns
    if c not in ["failure", "failure_id"]
]

print(features)

print("\nTotal sensor features:", len(features))
