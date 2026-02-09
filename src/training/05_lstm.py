"""
Model 4: Small GRU network (lighter alternative to LSTM)
Uses a sequence of the last 6 hours to predict PM2.5 1 hour ahead.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Load data
train = pd.read_csv("../../Data/processed/train.csv", index_col="datetime_utc", parse_dates=True)
test = pd.read_csv("../../Data/processed/test.csv", index_col="datetime_utc", parse_dates=True)

target = "target_pm25_1h"
features = [c for c in train.columns if c != target]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train[features])
X_test_scaled = scaler.transform(test[features])
y_train = train[target].values
y_test = test[target].values

# Create sequences (last 6 hours as input)
SEQ_LEN = 6

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, SEQ_LEN)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, SEQ_LEN)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train_seq)
y_train_t = torch.FloatTensor(y_train_seq)
X_test_t = torch.FloatTensor(X_test_seq)

train_ds = TensorDataset(X_train_t, y_train_t)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)


# GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()


model = GRUModel(input_size=len(features))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Train
EPOCHS = 30
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_dl:
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(train_dl)
        print(f"  Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    y_pred = model(X_test_t).numpy()

print("\nGRU:")
print(f"  MAE:  {mean_absolute_error(y_test_seq, y_pred):.3f}")
print(f"  RMSE: {root_mean_squared_error(y_test_seq, y_pred):.3f}")
print(f"  R2:   {r2_score(y_test_seq, y_pred):.3f}")

# Save model and scaler
torch.save(model.state_dict(), "../../Data/models/gru_model.pt")
with open("../../Data/models/gru_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
