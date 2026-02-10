"""
Alert System: Escalation levels based on predicted PM2.5 (WHO thresholds)
- Green:  < 15 µg/m³ → no action
- Yellow: 15-25 µg/m³ → speed limit 30 km/h
- Red:    > 25 µg/m³  → reroute traffic
"""

import numpy as np
import pandas as pd
import onnxruntime as ort

# Load test data
test = pd.read_csv("../../Data/processed/test.csv", index_col="datetime_utc", parse_dates=True)

target = "target_pm25_1h"
drop_features = [
    "hour", "hour_sin", "hour_cos",
    "month", "month_sin", "month_cos",
    "day_of_week", "dow_sin", "dow_cos",
    "is_weekend", "is_rushhour",
]
features = [c for c in test.columns if c != target and c not in drop_features]
X_test = test[features].values.astype(np.float32)

# Run inference with best model (Linear Regression ONNX)
sess = ort.InferenceSession("../../Data/models/onnx/linear_regression.onnx")
y_pred = sess.run(None, {"X": X_test})[0].flatten()

# Classify alert levels
YELLOW_THRESHOLD = 15  # WHO 24h guideline
RED_THRESHOLD = 25     # WHO interim target 4

def classify(value):
    if value > RED_THRESHOLD:
        return "RED"
    elif value > YELLOW_THRESHOLD:
        return "YELLOW"
    return "GREEN"

alerts = pd.Series([classify(v) for v in y_pred], index=test.index, name="alert")

# Distribution
counts = alerts.value_counts()
total = len(alerts)
print("Alert distribution:")
for level in ["GREEN", "YELLOW", "RED"]:
    n = counts.get(level, 0)
    print(f"  {level:<6s}: {n:>5d} ({n/total*100:.1f}%)")

# Show alert transitions (when level changes)
transitions = alerts[alerts != alerts.shift()]
print(f"\nAlert transitions: {len(transitions)}")
print(f"\nLast 10 transitions:")
for ts, level in transitions.tail(10).items():
    print(f"  {ts}  -> {level}")
