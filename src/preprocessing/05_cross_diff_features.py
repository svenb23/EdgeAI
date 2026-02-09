"""
Step 5: Create cross-pollutant and difference features
- PM2.5 / PM10 ratio (combustion vs abrasion indicator)
- PM2.5 difference (rate of change)
"""

import pandas as pd

# Load data with rolling features
df = pd.read_csv("../../Data/processed/04_rolling_features.csv", index_col="datetime_utc", parse_dates=True)

# Cross-pollutant ratio
df["pm25_pm10_ratio"] = df["pm25"] / df["pm10"].replace(0, float("nan"))

# Difference features (rate of change)
df["pm25_diff_1h"] = df["pm25"] - df["pm25_lag_1"]
df["pm25_diff_3h"] = df["pm25"] - df["pm25"].shift(3)

# Drop rows with NaN
df = df.dropna()

# Save
df.to_csv("../../Data/processed/05_final_features.csv")
