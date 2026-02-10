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
    ("src/preprocessing/01_reshape_data.py",      "Step 1/10: Reshaping raw data"),
    ("src/preprocessing/02_time_features.py",      "Step 2/10: Time features"),
    ("src/preprocessing/03_lag_features.py",       "Step 3/10: Lag features"),
    ("src/preprocessing/04_rolling_features.py",   "Step 4/10: Rolling features"),
    ("src/preprocessing/05_cross_diff_features.py","Step 5/10: Cross-pollutant features"),
    # Training
    ("src/training/01_train_test_split.py",        "Step 6/10: Train/test split"),
    ("src/training/02_linear_regression.py",       "Step 7a/10: Linear Regression"),
    ("src/training/03_random_forest.py",           "Step 7b/10: Random Forest"),
    ("src/training/04_gradient_boosting.py",       "Step 7c/10: Gradient Boosting"),
    ("src/training/05_lstm.py",                    "Step 7d/10: GRU"),
    # Analysis
    ("src/analysis/06_feature_importance.py",      "Step 8/10: Feature importance"),
    ("src/analysis/07_reduced_features.py",        "Step 9a/10: Reduced feature retraining"),
    ("src/analysis/08_onnx_export.py",             "Step 9b/10: ONNX export"),
    ("src/analysis/09_benchmark.py",               "Step 9c/10: Benchmark"),
    ("src/analysis/10_alert_system.py",            "Step 10/10: Alert system"),
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
