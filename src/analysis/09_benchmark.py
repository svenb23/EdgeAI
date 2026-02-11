"""
Benchmark: Inference time and model size for PM2.5 and NO2 ONNX models.
"""

import os
import time
import numpy as np
import pandas as pd
import onnxruntime as ort

# Load data
test = pd.read_csv("../../Data/processed/test.csv", index_col="datetime_utc", parse_dates=True)

drop_features = [
    "hour", "hour_sin", "hour_cos",
    "month", "month_sin", "month_cos",
    "day_of_week", "dow_sin", "dow_cos",
    "is_weekend", "is_rushhour",
]
features = [c for c in test.columns if not c.startswith("target_") and c not in drop_features]
X_test = test[features].values.astype(np.float32)

N_RUNS = 100

print(f"{'Model':<30} {'Size':>10} {'Inference (single)':>20} {'Inference (batch)':>20}")
print("-" * 83)

for target in ["pm25", "no2"]:
    path = f"../../Data/models/onnx/{target}/linear_regression.onnx"
    size_kb = os.path.getsize(path) / 1024
    sess = ort.InferenceSession(path)

    # Single inference
    single = X_test[:1]
    times = []
    for _ in range(N_RUNS):
        start = time.perf_counter()
        sess.run(None, {"X": single})
        times.append(time.perf_counter() - start)
    avg_single_ms = np.mean(times) * 1000

    # Batch inference
    start = time.perf_counter()
    sess.run(None, {"X": X_test})
    batch_ms = (time.perf_counter() - start) * 1000

    print(f"linear_regression ({target}){' '*(10-len(target))} {size_kb:>8.1f} KB {avg_single_ms:>17.3f} ms {batch_ms:>17.1f} ms")
