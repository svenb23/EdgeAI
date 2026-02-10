"""
Step 1: Train/Test split
- Targets: PM2.5 and NO2 value 1 hour ahead
- Split: first 80% training, last 20% test (chronological)
- Save splits as separate CSV files
"""

import pandas as pd

# Load final features
df = pd.read_csv("../../Data/processed/05_final_features.csv", index_col="datetime_utc", parse_dates=True)

# Create targets: 1 hour ahead
df["target_pm25_1h"] = df["pm25"].shift(-1)
df["target_no2_1h"] = df["no2"].shift(-1)
df = df.dropna()

# Chronological split (80/20)
split_idx = int(len(df) * 0.8)
df_train = df.iloc[:split_idx]
df_test = df.iloc[split_idx:]

# Save
df_train.to_csv("../../Data/processed/train.csv")
df_test.to_csv("../../Data/processed/test.csv")
