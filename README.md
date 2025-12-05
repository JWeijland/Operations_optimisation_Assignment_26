# Operations Optimisation Assignment

Aircraft Landing Scheduling Problem - Implementatie voor AE vakken

## 🎯 Snelstart

### Methode 1: Met het run script (Makkelijkst!)

```bash
cd "/Users/jelleweijland/Documents/AE Vakken/Operations_Assignment/Operations_optimisation_Assignment_26"
./RUN.sh
```

### Methode 2: Handmatig

```bash
# Stap 1: Ga naar deze directory
cd "/Users/jelleweijland/Documents/AE Vakken/Operations_Assignment/Operations_optimisation_Assignment_26"

# Stap 2: Activeer virtual environment
source Operations_Guusje/bin/activate

# Stap 3: Run het programma
python aircraft_landing_scheduling/code/main.py
```

## 📂 Project Structuur

```
Operations_optimisation_Assignment_26/
├── RUN.sh                              # ← Run dit script!
├── Operations_Guusje/                  # Virtual environment
├── aircraft_landing_scheduling/        # Hoofdproject
│   ├── code/                          # Python source code
│   │   ├── main.py                    # Entry point met menu
│   │   ├── model.py                   # MIP model
│   │   ├── heuristic.py               # Greedy heuristic
│   │   ├── solver.py                  # Solver framework
│   │   ├── data_loader.py             # Data inlezen
│   │   ├── visualization.py           # Grafieken maken
│   │   ├── utils.py                   # Helper functies
│   │   └── schiphol_scenarios.py      # Realistische scenarios
│   ├── data/                          # Problem instances
│   ├── requirements.txt               # Python dependencies
│   ├── README.md                      # Gedetailleerde documentatie
│   └── INSTALL.md                     # Installatie instructies
├── results/                            # ← Alle output komt hier!
│   ├── scenario_1runway_light/
│   ├── scenario_1runway_heavy/
│   └── ...
└── support_documents/                  # Referentie documenten
```

## 🎮 Interactief Menu

Bij het starten krijg je een menu:

```
======================================================================
AIRCRAFT LANDING SCHEDULING - SCENARIO MENU
======================================================================

Available Scenarios:

  1. Schiphol 1 Runway Light     (20 aircraft, 1 runway)
  2. Schiphol 1 Runway Heavy     (35 aircraft, 1 runway)
  3. Schiphol 2 Runways Medium   (30 aircraft, 2 runways)
  4. Schiphol 3 Runways Heavy    (50 aircraft, 3 runways)
  5. Custom Sample Instance
  6. Load Custom File

Enter your choice (1-6): _
```

## 📊 Output

Elk scenario genereert:
- **4 PNG visualisaties**: Gantt charts, vergelijking, cost breakdown
- **2 CSV bestanden**: Gedetailleerde oplossingen (heuristic + optimal)
- **2 Excel bestanden**: Geformatteerde oplossingen

Alles wordt opgeslagen in: `results/scenario_naam/`

## ⏱️ Geschatte looptijden

| Scenario | Aircraft | Runways | Tijd |
|----------|----------|---------|------|
| 1. Light | 20 | 1 | ~10 seconden |
| 2. Heavy | 35 | 1 | ~30 seconden |
| 3. Medium | 30 | 2 | ~45 seconden |
| 4. Heavy | 50 | 3 | ~1-3 minuten |

## 🔧 Command-line opties

```bash
# Specifiek scenario met custom tijd
./RUN.sh --instance data/schiphol_1runway_light.txt --runways 1 --time-limit 600

# Alleen heuristic (snel)
./RUN.sh --instance sample --time-limit 5

# Help
./RUN.sh --help
```

## 📚 Meer informatie

- Gedetailleerde documentatie: `aircraft_landing_scheduling/README.md`
- Installatie instructies: `aircraft_landing_scheduling/INSTALL.md`
- Support documenten: `support_documents/`

## 🐛 Troubleshooting

**Virtual environment niet gevonden:**
```bash
python3 -m venv Operations_Guusje
source Operations_Guusje/bin/activate
pip install -r aircraft_landing_scheduling/requirements.txt
```

**Module niet gevonden:**
```bash
source Operations_Guusje/bin/activate
pip install -r aircraft_landing_scheduling/requirements.txt
```

**Results folder op verkeerde plek:**
→ Zorg dat je ALTIJD vanuit deze directory (Operations_optimisation_Assignment_26) runt!

## 👥 Team

AE Vakken - Operations Assignment 2024
