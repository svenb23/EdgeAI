# EdgeAI – Dynamische Verkehrssteuerung bei Schadstoffbelastung

Edge-AI-System zur Vorhersage der Luftqualität (PM2.5, NO2) an einer Straßenstation in Hamburg. Bei erhöhter Schadstoffbelastung werden automatisch Maßnahmen ausgelöst (Geschwindigkeitsbegrenzung, Umleitung), um Anwohner, insbesondere Schulkinder, zu schützen.

## Setup

```bash
# Repository klonen
git clone https://github.com/svenb23/EdgeAI.git
cd EdgeAI

# Virtuelle Umgebung erstellen und aktivieren
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Abhängigkeiten installieren
pip install -r requirements.txt
```

## Komplette Pipeline ausführen

Ein einziger Befehl führt die gesamte Pipeline aus – von den Rohdaten bis zum Alert-System:

```bash
python run_all.py
```

Die Pipeline durchläuft folgende Schritte:

1. Daten-Preprocessing (Reshape, Zeit-, Lag-, Rolling- und Cross-Features)
2. Train/Test-Split (chronologisch 80/20, 1h-Ahead Targets für PM2.5 und NO2)
3. Modelltraining PM2.5 (Linear Regression, Random Forest, Gradient Boosting, GRU)
4. Modelltraining NO2 (Linear Regression, Random Forest, Gradient Boosting, GRU)
5. Feature Importance Analyse (je Schadstoff)
6. Retraining mit reduziertem Feature-Set (28 → 17 Features)
7. ONNX-Export der besten Modelle (PM2.5 + NO2)
8. Benchmark (Inferenzzeit, Modellgröße)
9. Kombiniertes Alert-System (Worst-Case aus PM2.5 + NO2)
10. Edge-Inference-Demo (End-to-End: Sensor → Features → Vorhersage → Ampel)

## Projektstruktur

```
EdgeAI/
├── run_all.py                    # Komplette Pipeline in einem Schritt
├── src/
│   ├── preprocessing/            # Datenvorverarbeitung (5 Schritte)
│   │   ├── 01_reshape_data.py
│   │   ├── 02_time_features.py
│   │   ├── 03_lag_features.py
│   │   ├── 04_rolling_features.py
│   │   └── 05_cross_diff_features.py
│   ├── training/                 # Modelltraining
│   │   ├── 01_train_test_split.py
│   │   ├── pm25/                 # PM2.5 Modelle
│   │   │   ├── 02_linear_regression.py
│   │   │   ├── 03_random_forest.py
│   │   │   ├── 04_gradient_boosting.py
│   │   │   └── 05_gru.py
│   │   └── no2/                  # NO2 Modelle
│   │       ├── 02_linear_regression.py
│   │       ├── 03_random_forest.py
│   │       ├── 04_gradient_boosting.py
│   │       └── 05_gru.py
│   ├── analysis/                 # Analyse & Edge-Optimierung
│   │   ├── pm25/                 # PM2.5 Feature Importance & Reduced
│   │   │   ├── 06_feature_importance.py
│   │   │   └── 07_reduced_features.py
│   │   ├── no2/                  # NO2 Feature Importance & Reduced
│   │   │   ├── 06_feature_importance.py
│   │   │   └── 07_reduced_features.py
│   │   ├── 08_onnx_export.py     # ONNX-Export (PM2.5 + NO2)
│   │   ├── 09_benchmark.py       # Inferenz-Benchmark
│   │   └── 10_alert_system.py    # Kombiniertes Alert-System
│   └── inference/                # Edge-Inference
│       └── edge_inference.py     # End-to-End Edge-Komponente
├── Data/
│   ├── raw/                      # Rohdaten (OpenAQ)
│   ├── processed/                # Vorverarbeitete Daten & Train/Test
│   └── models/                   # Gespeicherte Modelle
│       ├── pm25/                 # PM2.5 Modelle (.pkl, .pt)
│       ├── no2/                  # NO2 Modelle (.pkl, .pt)
│       └── onnx/                 # ONNX-Modelle
│           ├── pm25/
│           └── no2/
├── requirements.txt              # Python-Abhängigkeiten
└── README.md
```

## Datenquelle

Stündliche Luftqualitätsmessungen der Station **Hamburg Habichtstraße** (OpenAQ, Location ID 3010) für das Jahr 2025.
