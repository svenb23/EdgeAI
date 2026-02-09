"""
Step 4: Create rolling statistics
- Rolling mean for PM2.5 (3h, 6h, 24h)
- Rolling standard deviation for PM2.5 (3h)
"""

import pandas as pd

# Load data with lag features
df = pd.read_csv("../../Data/processed/03_lag_features.csv", index_col="datetime_utc", parse_dates=True)

# Rolling mean for PM2.5
df["pm25_roll_mean_3h"] = df["pm25"].rolling(window=3).mean()
df["pm25_roll_mean_6h"] = df["pm25"].rolling(window=6).mean()
df["pm25_roll_mean_24h"] = df["pm25"].rolling(window=24).mean()

# Rolling standard deviation for PM2.5
df["pm25_roll_std_3h"] = df["pm25"].rolling(window=3).std()

# Drop rows with NaN from rolling
df = df.dropna()

# Save
df.to_csv("../../Data/processed/04_rolling_features.csv")
