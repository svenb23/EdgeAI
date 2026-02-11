"""
Combined Alert System: PM2.5 + NO2 (worst-case escalation)
- Green:  PM2.5 < 15 AND NO2 < 40  → no action
- Yellow: PM2.5 15-25 OR NO2 40-80 → speed limit 30 km/h
- Red:    PM2.5 > 25 OR NO2 > 80   → reroute traffic
"""

import numpy as np
import pandas as pd
import onnxruntime as ort

# Load test data
test = pd.read_csv("../../Data/processed/test.csv", index_col="datetime_utc", parse_dates=True)

drop_features = [
    "hour", "hour_sin", "hour_cos",
    "month", "month_sin", "month_cos",
    "day_of_week", "dow_sin", "dow_cos",
    "is_weekend", "is_rushhour",
]
features = [c for c in test.columns if not c.startswith("target_") and c not in drop_features]
X_test = test[features].values.astype(np.float32)

# Load both models
sess_pm25 = ort.InferenceSession("../../Data/models/onnx/pm25/linear_regression.onnx")
sess_no2 = ort.InferenceSession("../../Data/models/onnx/no2/linear_regression.onnx")

y_pm25 = sess_pm25.run(None, {"X": X_test})[0].flatten()
y_no2 = sess_no2.run(None, {"X": X_test})[0].flatten()

# Classify per pollutant, then take worst case
def classify(pm25, no2):
    if pm25 > 25 or no2 > 80:
        return "RED"
    elif pm25 > 15 or no2 > 40:
        return "YELLOW"
    return "GREEN"

alerts = pd.Series(
    [classify(p, n) for p, n in zip(y_pm25, y_no2)],
    index=test.index, name="alert"
)

# Distribution
counts = alerts.value_counts()
total = len(alerts)
print("Alert distribution (combined PM2.5 + NO2):")
for level in ["GREEN", "YELLOW", "RED"]:
    n = counts.get(level, 0)
    print(f"  {level:<6s}: {n:>5d} ({n/total*100:.1f}%)")

# Transitions
transitions = alerts[alerts != alerts.shift()]
print(f"\nAlert transitions: {len(transitions)}")
print(f"\nLast 10 transitions:")
for ts, level in transitions.tail(10).items():
    pm25_val = y_pm25[test.index.get_loc(ts)]
    no2_val = y_no2[test.index.get_loc(ts)]
    print(f"  {ts}  -> {level} (PM2.5: {pm25_val:.1f}, NO2: {no2_val:.1f})")
