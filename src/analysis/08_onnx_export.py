"""
ONNX Export for best models (reduced feature set) – PM2.5 and NO2
Trains Linear Regression for each target, exports to ONNX, verifies.
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

# Load data
train = pd.read_csv("../../Data/processed/train.csv", index_col="datetime_utc", parse_dates=True)
test = pd.read_csv("../../Data/processed/test.csv", index_col="datetime_utc", parse_dates=True)

drop_features = [
    "hour", "hour_sin", "hour_cos",
    "month", "month_sin", "month_cos",
    "day_of_week", "dow_sin", "dow_cos",
    "is_weekend", "is_rushhour",
]
features = [c for c in train.columns if not c.startswith("target_") and c not in drop_features]

initial_type = [("X", FloatTensorType([None, len(features)]))]

targets = {
    "pm25": "target_pm25_1h",
    "no2": "target_no2_1h",
}

for name, target_col in targets.items():
    onnx_dir = f"../../Data/models/onnx/{name}"
    os.makedirs(onnx_dir, exist_ok=True)

    X_train, y_train = train[features], train[target_col]
    X_test, y_test = test[features], test[target_col]

    model = LinearRegression()
    model.fit(X_train, y_train)

    onnx_model = convert_sklearn(model, initial_types=initial_type)
    path = f"{onnx_dir}/linear_regression.onnx"
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Verify
    sess = ort.InferenceSession(path)
    y_onnx = sess.run(None, {"X": X_test.values.astype(np.float32)})[0].flatten()
    size_kb = os.path.getsize(path) / 1024
    print(f"{name}: MAE={mean_absolute_error(y_test, y_onnx):.3f}, size={size_kb:.1f} KB")
