"""
Model 2: Random Forest
"""

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Load data
train = pd.read_csv("../../Data/processed/train.csv", index_col="datetime_utc", parse_dates=True)
test = pd.read_csv("../../Data/processed/test.csv", index_col="datetime_utc", parse_dates=True)

target = "target_pm25_1h"
features = [c for c in train.columns if c != target]

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

# Train
model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Random Forest:")
print(f"  MAE:  {mean_absolute_error(y_test, y_pred):.3f}")
print(f"  RMSE: {root_mean_squared_error(y_test, y_pred):.3f}")
print(f"  R2:   {r2_score(y_test, y_pred):.3f}")

# Save model
with open("../../Data/models/random_forest.pkl", "wb") as f:
    pickle.dump(model, f)
