"""
EdgeAI – Complete Pipeline
Runs all steps from raw data to ONNX benchmark and alert system.
Usage: python run_all.py
"""

import subprocess
import sys
import time
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    # Preprocessing
    ("src/preprocessing/01_reshape_data.py",         "Step 1/13: Reshaping raw data"),
    ("src/preprocessing/02_time_features.py",         "Step 2/13: Time features"),
    ("src/preprocessing/03_lag_features.py",          "Step 3/13: Lag features"),
    ("src/preprocessing/04_rolling_features.py",      "Step 4/13: Rolling features"),
    ("src/preprocessing/05_cross_diff_features.py",   "Step 5/13: Cross-pollutant features"),
    # Training
    ("src/training/01_train_test_split.py",           "Step 6/13: Train/test split"),
    # PM2.5 models
    ("src/training/pm25/02_linear_regression.py",     "Step 7a/13: Linear Regression (PM2.5)"),
    ("src/training/pm25/03_random_forest.py",         "Step 7b/13: Random Forest (PM2.5)"),
    ("src/training/pm25/04_gradient_boosting.py",     "Step 7c/13: Gradient Boosting (PM2.5)"),
    ("src/training/pm25/05_gru.py",                    "Step 7d/13: GRU (PM2.5)"),
    # NO2 models
    ("src/training/no2/02_linear_regression.py",      "Step 7e/13: Linear Regression (NO2)"),
    ("src/training/no2/03_random_forest.py",          "Step 7f/13: Random Forest (NO2)"),
    ("src/training/no2/04_gradient_boosting.py",      "Step 7g/13: Gradient Boosting (NO2)"),
    ("src/training/no2/05_gru.py",                    "Step 7h/13: GRU (NO2)"),
    # PM2.5 analysis
    ("src/analysis/pm25/06_feature_importance.py",    "Step 8a/13: Feature importance (PM2.5)"),
    ("src/analysis/pm25/07_reduced_features.py",      "Step 8b/13: Reduced features (PM2.5)"),
    # NO2 analysis
    ("src/analysis/no2/06_feature_importance.py",     "Step 9a/13: Feature importance (NO2)"),
    ("src/analysis/no2/07_reduced_features.py",       "Step 9b/13: Reduced features (NO2)"),
    # Shared
    ("src/analysis/08_onnx_export.py",                "Step 10/13: ONNX export (PM2.5 + NO2)"),
    ("src/analysis/09_benchmark.py",                  "Step 11/13: Benchmark"),
    ("src/analysis/10_alert_system.py",               "Step 12/13: Alert system (combined)"),
    ("src/inference/edge_inference.py",               "Step 13/13: Edge inference demo"),
]


def run_step(script, description):
    print(f"\n{'='*60}")
    print(f" {description}")
    print(f"{'='*60}")

    script_path = os.path.join(ROOT, script.replace("/", os.sep))
    script_dir = os.path.dirname(script_path)

    result = subprocess.run([sys.executable, script_path], cwd=script_dir)

    if result.returncode != 0:
        print(f"\nERROR: {script} failed (exit code {result.returncode})")
        sys.exit(1)


if __name__ == "__main__":
    start = time.time()
    print("EdgeAI – Running complete pipeline")
    print(f"Python: {sys.executable}")

    for script, desc in STEPS:
        run_step(script, desc)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f" Pipeline complete! ({elapsed:.1f}s)")
    print(f"{'='*60}")
