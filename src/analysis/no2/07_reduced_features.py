"""
Retrain all models with reduced feature set – NO2
Drops time-based features (same as PM2.5 analysis showed them as irrelevant).
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
train = pd.read_csv("../../../Data/processed/train.csv", index_col="datetime_utc", parse_dates=True)
test = pd.read_csv("../../../Data/processed/test.csv", index_col="datetime_utc", parse_dates=True)

target = "target_no2_1h"

# Features to drop (time-based, low importance)
drop_features = [
    "hour", "hour_sin", "hour_cos",
    "month", "month_sin", "month_cos",
    "day_of_week", "dow_sin", "dow_cos",
    "is_weekend", "is_rushhour",
]

all_features = [c for c in train.columns if not c.startswith("target_")]
reduced_features = [c for c in all_features if c not in drop_features]

print(f"Features: {len(all_features)} -> {len(reduced_features)}")

X_train_all, y_train = train[all_features], train[target]
X_test_all, y_test = test[all_features], test[target]
X_train, X_test = train[reduced_features], test[reduced_features]

def evaluate(name, y_true, y_pred_all, y_pred_red):
    mae_all = mean_absolute_error(y_true, y_pred_all)
    mae_red = mean_absolute_error(y_true, y_pred_red)
    rmse_all = root_mean_squared_error(y_true, y_pred_all)
    rmse_red = root_mean_squared_error(y_true, y_pred_red)
    r2_all = r2_score(y_true, y_pred_all)
    r2_red = r2_score(y_true, y_pred_red)
    print(f"\n{name}:")
    print(f"  MAE:  {mae_red:.3f}  (baseline: {mae_all:.3f}, diff: {mae_red - mae_all:+.3f})")
    print(f"  RMSE: {rmse_red:.3f}  (baseline: {rmse_all:.3f}, diff: {rmse_red - rmse_all:+.3f})")
    print(f"  R2:   {r2_red:.3f}  (baseline: {r2_all:.3f}, diff: {r2_red - r2_all:+.3f})")

# 1. Linear Regression
lr_all = LinearRegression().fit(X_train_all, y_train)
lr_red = LinearRegression().fit(X_train, y_train)
evaluate("Linear Regression", y_test, lr_all.predict(X_test_all), lr_red.predict(X_test))

# 2. Random Forest
rf_all = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1).fit(X_train_all, y_train)
rf_red = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1).fit(X_train, y_train)
evaluate("Random Forest", y_test, rf_all.predict(X_test_all), rf_red.predict(X_test))

# 3. Gradient Boosting
gb_all = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42).fit(X_train_all, y_train)
gb_red = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42).fit(X_train, y_train)
evaluate("Gradient Boosting", y_test, gb_all.predict(X_test_all), gb_red.predict(X_test))

# 4. GRU
scaler_all = StandardScaler()
scaler_red = StandardScaler()
X_tr_sc_all = scaler_all.fit_transform(X_train_all)
X_te_sc_all = scaler_all.transform(X_test_all)
X_tr_sc_red = scaler_red.fit_transform(X_train)
X_te_sc_red = scaler_red.transform(X_test)

SEQ_LEN = 6

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def train_gru(X_tr, y_tr, n_features):
    X_seq, y_seq = create_sequences(X_tr, y_tr, SEQ_LEN)
    ds = TensorDataset(torch.FloatTensor(X_seq), torch.FloatTensor(y_seq))
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    class GRUModel(nn.Module):
        def __init__(self, input_size, hidden_size=32):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(out[:, -1, :]).squeeze()

    model = GRUModel(n_features)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    for _ in range(30):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return model

gru_all = train_gru(X_tr_sc_all, y_train.values, len(all_features))
gru_red = train_gru(X_tr_sc_red, y_train.values, len(reduced_features))

_, y_test_seq = create_sequences(X_te_sc_all, y_test.values, SEQ_LEN)
gru_all.eval()
gru_red.eval()
with torch.no_grad():
    X_te_seq_all = torch.FloatTensor(np.array([X_te_sc_all[i-SEQ_LEN:i] for i in range(SEQ_LEN, len(X_te_sc_all))]))
    X_te_seq_red = torch.FloatTensor(np.array([X_te_sc_red[i-SEQ_LEN:i] for i in range(SEQ_LEN, len(X_te_sc_red))]))
    pred_all = gru_all(X_te_seq_all).numpy()
    pred_red = gru_red(X_te_seq_red).numpy()
evaluate("GRU", y_test_seq, pred_all, pred_red)
