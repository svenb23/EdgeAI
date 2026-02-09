"""
Step 3: Create lag features
- Previous values for PM2.5 (t-1, t-2, t-3)
- Previous values for NO2 and CO (t-1)
"""

import pandas as pd

# Load data with time features
df = pd.read_csv("../../Data/processed/02_time_features.csv", index_col="datetime_utc", parse_dates=True)

# PM2.5 lag features (target variable history)
df["pm25_lag_1"] = df["pm25"].shift(1)
df["pm25_lag_2"] = df["pm25"].shift(2)
df["pm25_lag_3"] = df["pm25"].shift(3)

# NO2 and CO lag features (traffic indicators)
df["no2_lag_1"] = df["no2"].shift(1)
df["co_lag_1"] = df["co"].shift(1)

# PM10 lag feature
df["pm10_lag_1"] = df["pm10"].shift(1)

# Drop rows with NaN from shifting
df = df.dropna()

# Save
df.to_csv("../../Data/processed/03_lag_features.csv")
