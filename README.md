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

## Projektstruktur

```
EdgeAI/
├── src/                  # Quellcode (Preprocessing, Training, Inference)
├── Data/
│   ├── raw/              # Rohdaten (OpenAQ)
│   └── processed/        # Vorverarbeitete Daten
├── requirements.txt      # Python-Abhängigkeiten
└── README.md
```

## Datenquelle

Stündliche Luftqualitätsmessungen der Station **Hamburg Habichtstraße** (OpenAQ, Location ID 3010) für das Jahr 2025.
