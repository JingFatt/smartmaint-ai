import sys

import httpx
import pandas as pd

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/predict"

FEATURES = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature",
            "Motor_current", "COMP", "DV_eletric", "Towers", "MPG", "LPS",
            "Pressure_switch", "Oil_level", "Caudal_impulses"]

df = pd.read_parquet("data/processed/metropt3_flagged.parquet").sort_values("timestamp")
cases = {
    "normal (Aug 2020)": df[df["clean_normal"] & (df["timestamp"] >= "2020-08-10")].iloc[:180],
    "failure 3 (mid)": df[df["failure_id"] == 3].iloc[3000:3180],
    "failure 4 (mid)": df[df["failure_id"] == 4].iloc[900:1080],
}
for name, rows in cases.items():
    body = {"equipment_id": name, "readings": rows[FEATURES].to_dict("records")}
    r = httpx.post(URL, json=body, timeout=30)
    d = r.json()
    print(f"{name:20s} IF alarm={d['if_alarm']} (flagged {d['flagged_fraction']:.2f}) | "
          f"rule alarm={d['rule_alarm']} (load {d['load_fraction']:.2f})")