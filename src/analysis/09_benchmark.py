"""
Benchmark: Inference time, model size, and memory usage for all ONNX models.
"""

import os
import time
import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler

# Load data
test = pd.read_csv("../../Data/processed/test.csv", index_col="datetime_utc", parse_dates=True)
train = pd.read_csv("../../Data/processed/train.csv", index_col="datetime_utc", parse_dates=True)

target = "target_pm25_1h"
drop_features = [
    "hour", "hour_sin", "hour_cos",
    "month", "month_sin", "month_cos",
    "day_of_week", "dow_sin", "dow_cos",
    "is_weekend", "is_rushhour",
]
features = [c for c in test.columns if c != target and c not in drop_features]

X_test = test[features].values.astype(np.float32)

# GRU needs scaled sequences
scaler = StandardScaler()
scaler.fit(train[features])
X_test_scaled = scaler.transform(test[features]).astype(np.float32)

SEQ_LEN = 6
X_test_seq = np.array([X_test_scaled[i - SEQ_LEN:i] for i in range(SEQ_LEN, len(X_test_scaled))])

onnx_dir = "../../Data/models/onnx"
N_RUNS = 100

models = {
    "linear_regression": {"input": X_test},
    "random_forest": {"input": X_test},
    "gradient_boosting": {"input": X_test},
    "gru": {"input": X_test_seq},
}

print(f"{'Model':<22} {'Size':>10} {'Inference (single)':>20} {'Inference (batch)':>20}")
print("-" * 75)

for name, cfg in models.items():
    path = f"{onnx_dir}/{name}.onnx"
    size_kb = os.path.getsize(path) / 1024
    sess = ort.InferenceSession(path)
    X = cfg["input"]

    # Single inference (1 sample, averaged over N_RUNS)
    single = X[:1]
    times = []
    for _ in range(N_RUNS):
        start = time.perf_counter()
        sess.run(None, {"X": single})
        times.append(time.perf_counter() - start)
    avg_single_ms = np.mean(times) * 1000

    # Batch inference (full test set)
    start = time.perf_counter()
    sess.run(None, {"X": X})
    batch_ms = (time.perf_counter() - start) * 1000

    print(f"{name:<22} {size_kb:>8.1f} KB {avg_single_ms:>17.3f} ms {batch_ms:>17.1f} ms")
