"""
Retrain all models with reduced feature set (top 17 features)
and compare against baseline (all 28 features).
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Load data
train = pd.read_csv("../../Data/processed/train.csv", index_col="datetime_utc", parse_dates=True)
test = pd.read_csv("../../Data/processed/test.csv", index_col="datetime_utc", parse_dates=True)

target = "target_pm25_1h"

# Features to drop (avg_rank >= 19)
drop_features = [
    "hour", "hour_sin", "hour_cos",
    "month", "month_sin", "month_cos",
    "day_of_week", "dow_sin", "dow_cos",
    "is_weekend", "is_rushhour",
]

all_features = [c for c in train.columns if c != target]
reduced_features = [c for c in all_features if c not in drop_features]

print(f"Features: {len(all_features)} -> {len(reduced_features)}")

X_train, y_train = train[reduced_features], train[target]
X_test, y_test = test[reduced_features], test[target]

# Baseline results (all 28 features)
baseline = {
    "Linear Regression": {"MAE": 0.929, "RMSE": 1.537, "R2": 0.930},
    "Random Forest":     {"MAE": 0.981, "RMSE": 1.578, "R2": 0.926},
    "Gradient Boosting": {"MAE": 0.990, "RMSE": 1.579, "R2": 0.926},
    "GRU":               {"MAE": 1.743, "RMSE": 2.493, "R2": 0.817},
}

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    b = baseline[name]
    print(f"\n{name}:")
    print(f"  MAE:  {mae:.3f}  (baseline: {b['MAE']:.3f}, diff: {mae - b['MAE']:+.3f})")
    print(f"  RMSE: {rmse:.3f}  (baseline: {b['RMSE']:.3f}, diff: {rmse - b['RMSE']:+.3f})")
    print(f"  R2:   {r2:.3f}  (baseline: {b['R2']:.3f}, diff: {r2 - b['R2']:+.3f})")

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
evaluate("Linear Regression", y_test, lr.predict(X_test))

# 2. Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
evaluate("Random Forest", y_test, rf.predict(X_test))

# 3. Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
evaluate("Gradient Boosting", y_test, gb.predict(X_test))

# 4. GRU
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

X_train_t = torch.FloatTensor(X_train_seq)
y_train_t = torch.FloatTensor(y_train_seq)
X_test_t = torch.FloatTensor(X_test_seq)

train_ds = TensorDataset(X_train_t, y_train_t)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze()

model = GRUModel(input_size=len(reduced_features))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(30):
    model.train()
    for X_batch, y_batch in train_dl:
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    y_pred_gru = model(X_test_t).numpy()
evaluate("GRU", y_test_seq, y_pred_gru)
