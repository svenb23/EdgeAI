"""
Edge Inference Pipeline
Simulates the full edge component: raw sensor data → features → ONNX prediction → alert level.
Predicts both PM2.5 and NO2, combined worst-case alert.
"""

import numpy as np
import pandas as pd
import onnxruntime as ort

BUFFER_SIZE = 24  # hours needed for pm25_roll_mean_24h

FEATURES = [
    "co", "no2", "pm10", "pm25",
    "pm25_lag_1", "pm25_lag_2", "pm25_lag_3",
    "no2_lag_1", "co_lag_1", "pm10_lag_1",
    "pm25_roll_mean_3h", "pm25_roll_mean_6h", "pm25_roll_mean_24h",
    "pm25_roll_std_3h",
    "pm25_pm10_ratio", "pm25_diff_1h", "pm25_diff_3h",
]


class EdgeInference:
    def __init__(self, pm25_model_path, no2_model_path):
        self.sess_pm25 = ort.InferenceSession(pm25_model_path)
        self.sess_no2 = ort.InferenceSession(no2_model_path)
        self.buffer = []

    def add_reading(self, pm25, pm10, no2, co):
        """Add a new sensor reading and return prediction + alert level."""
        self.buffer.append({"pm25": pm25, "pm10": pm10, "no2": no2, "co": co})

        if len(self.buffer) < BUFFER_SIZE:
            return None

        self.buffer = self.buffer[-BUFFER_SIZE:]
        features = self._compute_features()

        pred_pm25 = self.sess_pm25.run(None, {"X": features})[0].item()
        pred_no2 = self.sess_no2.run(None, {"X": features})[0].item()
        alert = self._classify(pred_pm25, pred_no2)

        return {
            "pred_pm25": round(pred_pm25, 2),
            "pred_no2": round(pred_no2, 2),
            "alert": alert,
        }

    def _compute_features(self):
        pm25_vals = [r["pm25"] for r in self.buffer]
        current = self.buffer[-1]

        return np.array([[
            current["co"],
            current["no2"],
            current["pm10"],
            current["pm25"],
            pm25_vals[-2],
            pm25_vals[-3],
            pm25_vals[-4],
            self.buffer[-2]["no2"],
            self.buffer[-2]["co"],
            self.buffer[-2]["pm10"],
            np.mean(pm25_vals[-3:]),
            np.mean(pm25_vals[-6:]),
            np.mean(pm25_vals[-24:]),
            np.std(pm25_vals[-3:], ddof=1),
            current["pm25"] / max(current["pm10"], 0.01),
            current["pm25"] - pm25_vals[-2],
            current["pm25"] - pm25_vals[-4],
        ]], dtype=np.float32)

    def _classify(self, pm25, no2):
        if pm25 > 25 or no2 > 80:
            return "RED"
        elif pm25 > 15 or no2 > 40:
            return "YELLOW"
        return "GREEN"


# --- Demo: simulate edge device with real test data ---
if __name__ == "__main__":
    test = pd.read_csv("../../Data/processed/test.csv", index_col="datetime_utc", parse_dates=True)

    edge = EdgeInference(
        "../../Data/models/onnx/pm25/linear_regression.onnx",
        "../../Data/models/onnx/no2/linear_regression.onnx",
    )

    results = []
    for ts, row in test.iterrows():
        result = edge.add_reading(row["pm25"], row["pm10"], row["no2"], row["co"])
        if result:
            results.append({"timestamp": ts, **result})

    df = pd.DataFrame(results).set_index("timestamp")

    # Summary
    alerts = df["alert"].value_counts()
    total = len(df)
    print(f"Processed {total} predictions\n")
    print("Alert distribution (combined PM2.5 + NO2):")
    for level in ["GREEN", "YELLOW", "RED"]:
        n = alerts.get(level, 0)
        print(f"  {level:<6s}: {n:>5d} ({n/total*100:.1f}%)")

    transitions = df[df["alert"] != df["alert"].shift()]
    print(f"\nLast 5 alert transitions:")
    for ts, row in transitions.tail(5).iterrows():
        print(f"  {ts}  -> {row['alert']} (PM2.5: {row['pred_pm25']}, NO2: {row['pred_no2']})")
