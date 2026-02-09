"""
Step 2: Create time-based features
- Hour, day of week, month
- Binary: weekend, rush hour
- Cyclical encoding (sin/cos)
"""

import pandas as pd
import numpy as np

# Load wide data
df = pd.read_csv("../../Data/processed/01_wide_data.csv", index_col="datetime_utc", parse_dates=True)

dt = df.index

# Basic time features
df["hour"] = dt.hour
df["day_of_week"] = dt.dayofweek
df["month"] = dt.month

# Binary features
df["is_weekend"] = (dt.dayofweek >= 5).astype(int)
df["is_rushhour"] = dt.hour.isin([7, 8, 9, 16, 17, 18]).astype(int)

# Cyclical encoding (sin/cos)
df["hour_sin"] = np.sin(2 * np.pi * dt.hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * dt.hour / 24)
df["month_sin"] = np.sin(2 * np.pi * dt.month / 12)
df["month_cos"] = np.cos(2 * np.pi * dt.month / 12)
df["dow_sin"] = np.sin(2 * np.pi * dt.dayofweek / 7)
df["dow_cos"] = np.cos(2 * np.pi * dt.dayofweek / 7)

# Save
df.to_csv("../../Data/processed/02_time_features.csv")
