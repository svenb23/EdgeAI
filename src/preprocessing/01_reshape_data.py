"""
Step 1: Reshape raw data
- Convert long format to wide format (one row per hour)
- Create complete hourly index
- Interpolate missing values
"""

import pandas as pd

# Load raw data
df_raw = pd.read_csv("../../Data/raw/hamburg_habichtstrasse_2025.csv")

# Long -> Wide format
df_raw["datetime_utc"] = pd.to_datetime(df_raw["datetime_utc"])

df_wide = df_raw.pivot_table(
    index="datetime_utc",
    columns="parameter",
    values="value",
    aggfunc="mean",
)
df_wide = df_wide.sort_index()
df_wide.columns.name = None

# Create complete hourly index for full year 2025
full_index = pd.date_range(
    start="2025-01-01 01:00:00",
    end="2025-12-31 23:00:00",
    freq="h",
    tz="UTC",
)
df_wide = df_wide.reindex(full_index)
df_wide.index.name = "datetime_utc"

# Interpolate missing values (max 6h gap)
df_wide = df_wide.interpolate(method="linear", limit=6)
df_wide = df_wide.dropna()

# Save
df_wide.to_csv("../../Data/processed/01_wide_data.csv")
