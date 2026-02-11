"""
Feature Importance Analysis – NO2
- Random Forest & Gradient Boosting importance (MDI)
- Permutation Importance (model-agnostic)
- Correlation with target
- Combined ranking across all methods
"""

import pandas as pd
import pickle
from sklearn.inspection import permutation_importance

# Load data
train = pd.read_csv("../../../Data/processed/train.csv", index_col="datetime_utc", parse_dates=True)
test = pd.read_csv("../../../Data/processed/test.csv", index_col="datetime_utc", parse_dates=True)

target = "target_no2_1h"
features = [c for c in train.columns if not c.startswith("target_")]

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

# Correlation with target
corr = train[features].corrwith(train[target]).abs().rename("correlation")

# Random Forest feature importance (MDI)
with open("../../../Data/models/no2/random_forest.pkl", "rb") as f:
    rf = pickle.load(f)
rf_imp = pd.Series(rf.feature_importances_, index=features, name="rf_importance")

# Gradient Boosting feature importance (MDI)
with open("../../../Data/models/no2/gradient_boosting.pkl", "rb") as f:
    gb = pickle.load(f)
gb_imp = pd.Series(gb.feature_importances_, index=features, name="gb_importance")

# Permutation Importance (Gradient Boosting, test set)
perm = permutation_importance(gb, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_imp = pd.Series(perm.importances_mean, index=features, name="perm_importance")

# Combine and rank
df_imp = pd.DataFrame({
    "correlation": corr,
    "rf_importance": rf_imp,
    "gb_importance": gb_imp,
    "perm_importance": perm_imp,
})

for col in df_imp.columns:
    df_imp[f"{col}_rank"] = df_imp[col].rank(ascending=False).astype(int)

rank_cols = [c for c in df_imp.columns if c.endswith("_rank")]
df_imp["avg_rank"] = df_imp[rank_cols].mean(axis=1)
df_imp = df_imp.sort_values("avg_rank")

# Print ranking
display_cols = ["correlation", "rf_importance", "gb_importance", "perm_importance", "avg_rank"]
print(df_imp[display_cols].to_string(float_format=lambda x: f"{x:.4f}"))

# Save
df_imp.to_csv("../../../Data/processed/feature_importance_no2.csv")
