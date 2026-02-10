"""
ONNX Export for all models (reduced feature set)
Trains each model, exports to ONNX, verifies with onnxruntime.
"""

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Load data
train = pd.read_csv("../../Data/processed/train.csv", index_col="datetime_utc", parse_dates=True)
test = pd.read_csv("../../Data/processed/test.csv", index_col="datetime_utc", parse_dates=True)

target = "target_pm25_1h"
drop_features = [
    "hour", "hour_sin", "hour_cos",
    "month", "month_sin", "month_cos",
    "day_of_week", "dow_sin", "dow_cos",
    "is_weekend", "is_rushhour",
]
all_features = [c for c in train.columns if c != target]
features = [c for c in all_features if c not in drop_features]

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

onnx_dir = "../../Data/models/onnx"
os.makedirs(onnx_dir, exist_ok=True)

# --- sklearn models ---
sklearn_models = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
}

initial_type = [("X", FloatTensorType([None, len(features)]))]

for name, model in sklearn_models.items():
    model.fit(X_train, y_train)
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    path = f"{onnx_dir}/{name}.onnx"
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Verify
    sess = ort.InferenceSession(path)
    y_onnx = sess.run(None, {"X": X_test.values.astype(np.float32)})[0].flatten()
    size_kb = os.path.getsize(path) / 1024
    print(f"{name}: MAE={mean_absolute_error(y_test, y_onnx):.3f}, size={size_kb:.1f} KB")

# --- GRU model ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

SEQ_LEN = 6

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, SEQ_LEN)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, SEQ_LEN)

train_ds = TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq))
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze()

gru = GRUModel(input_size=len(features))
optimizer = torch.optim.Adam(gru.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(30):
    gru.train()
    for X_batch, y_batch in train_dl:
        pred = gru(X_batch)
        loss = loss_fn(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Export GRU to ONNX
gru.eval()
dummy = torch.randn(1, SEQ_LEN, len(features))
gru_path = f"{onnx_dir}/gru.onnx"
torch.onnx.export(gru, dummy, gru_path, input_names=["X"], output_names=["output"],
                  dynamic_axes={"X": {0: "batch"}, "output": {0: "batch"}})

# Verify
sess = ort.InferenceSession(gru_path)
y_onnx_gru = sess.run(None, {"X": X_test_seq.astype(np.float32)})[0].flatten()
size_kb = os.path.getsize(gru_path) / 1024
print(f"gru: MAE={mean_absolute_error(y_test_seq, y_onnx_gru):.3f}, size={size_kb:.1f} KB")
