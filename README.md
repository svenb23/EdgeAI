# EdgeAI – Dynamische Verkehrssteuerung bei Schadstoffbelastung

Edge-AI-System zur Vorhersage der Luftqualität (PM2.5, PM10, NO2, CO) an einer Straßenstation in Hamburg. Bei erhöhter Schadstoffbelastung werden automatisch Maßnahmen ausgelöst (Geschwindigkeitsbegrenzung, Umleitung), um Anwohner – insbesondere Schulkinder – zu schützen.

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
2. Train/Test-Split (chronologisch 80/20, 1h-Ahead Target)
3. Modelltraining (Linear Regression, Random Forest, Gradient Boosting, GRU)
4. Feature Importance Analyse (Korrelation, MDI, Permutation Importance)
5. Retraining mit reduziertem Feature-Set (28 → 17 Features)
6. ONNX-Export aller Modelle
7. Benchmark (Inferenzzeit, Modellgröße)
8. Alert-System (Eskalationsstufen Grün/Gelb/Rot nach WHO-Schwellwerten)

## Projektstruktur

```
EdgeAI/
├── run_all.py                # Komplette Pipeline in einem Schritt
├── src/
│   ├── preprocessing/        # Datenvorverarbeitung (5 Schritte)
│   │   ├── 01_reshape_data.py
│   │   ├── 02_time_features.py
│   │   ├── 03_lag_features.py
│   │   ├── 04_rolling_features.py
│   │   └── 05_cross_diff_features.py
│   ├── training/             # Modelltraining
│   │   ├── 01_train_test_split.py
│   │   ├── 02_linear_regression.py
│   │   ├── 03_random_forest.py
│   │   ├── 04_gradient_boosting.py
│   │   └── 05_lstm.py
│   └── analysis/             # Analyse & Edge-Optimierung
│       ├── 06_feature_importance.py
│       ├── 07_reduced_features.py
│       ├── 08_onnx_export.py
│       ├── 09_benchmark.py
│       └── 10_alert_system.py
├── Data/
│   ├── raw/                  # Rohdaten (OpenAQ)
│   ├── processed/            # Vorverarbeitete Daten & Train/Test
│   └── models/               # Gespeicherte Modelle (.pkl, .pt, .onnx)
├── requirements.txt          # Python-Abhängigkeiten
└── README.md
```

## Datenquelle

Stündliche Luftqualitätsmessungen der Station **Hamburg Habichtstraße** (OpenAQ, Location ID 3010) für das Jahr 2025.
