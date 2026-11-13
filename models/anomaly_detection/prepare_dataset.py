
import pandas as pd
from pathlib import Path

# ==========================================
# 1. Load flagged dataset
# ==========================================

INPUT = Path("data/processed/metropt3_flagged.parquet")
OUTPUT = Path("data/processed")

df = pd.read_parquet(INPUT)
df = df.sort_values("timestamp").reset_index(drop=True)

# ==========================================
# 2. Define sensor features
# ==========================================

FEATURES = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses"
]

assert len(FEATURES) == 15
assert all(c in df.columns for c in FEATURES)

# ==========================================
# 3. Prepare clean normal records
# ==========================================

normal = df[df["clean_normal"]].copy()

# Chronological split:
# 70% training, 15% validation, 15% normal holdout

n = len(normal)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = normal.iloc[:train_end].copy()
validation = normal.iloc[train_end:val_end].copy()
normal_test = normal.iloc[val_end:].copy()

# ==========================================
# 4. Prepare fault evaluation records
# ==========================================

fault_test = df[
    (df["failure"] == 1) &
    (~df["frozen"]) &
    (~df["uncertain"])
].copy()

# Test dataset: normal holdout + confirmed failures
normal_test["target"] = 0
fault_test["target"] = 1

test = pd.concat(
    [normal_test, fault_test],
    ignore_index=True
).sort_values("timestamp")

# ==========================================
# 5. Export datasets
# ==========================================

train[FEATURES + ["timestamp"]].to_parquet(
    OUTPUT / "train_normal.parquet",
    index=False
)

validation[FEATURES + ["timestamp"]].to_parquet(
    OUTPUT / "validation.parquet",
    index=False
)

test[FEATURES + ["timestamp", "target", "failure_id"]].to_parquet(
    OUTPUT / "test.parquet",
    index=False
)

# ==========================================
# 6. Summary
# ==========================================

print("\n========== DATASET PREPARATION ==========")

print("Sensor features:", len(FEATURES))

print("\nTraining:", len(train))
print("Validation:", len(validation))
print("Normal test:", len(normal_test))
print("Fault test:", len(fault_test))
print("Total test:", len(test))

print("\nTest labels:")
print(test["target"].value_counts())

print("\nTime ranges:")
print("Train:", train["timestamp"].min(), "->", train["timestamp"].max())
print("Validation:", validation["timestamp"].min(), "->", validation["timestamp"].max())
print("Normal test:", normal_test["timestamp"].min(), "->", normal_test["timestamp"].max())

print("\nDataset preparation completed.")
