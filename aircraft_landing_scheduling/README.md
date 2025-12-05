# Aircraft Landing Scheduling Problem

Dit project implementeert optimale en heuristische oplossingen voor het Aircraft Landing Problem (ALP).

## 🚀 HOE TE RUNNEN

**BELANGRIJK: Run altijd vanuit de ROOT directory!**

```bash
# Methode 1: Met run script (MAKKELIJKST!)
cd "/Users/jelleweijland/Documents/AE Vakken/Operations_Assignment/Operations_optimisation_Assignment_26"
./RUN.sh

# Methode 2: Handmatig
cd "/Users/jelleweijland/Documents/AE Vakken/Operations_Assignment/Operations_optimisation_Assignment_26"
source Operations_Guusje/bin/activate
python aircraft_landing_scheduling/code/main.py
```

### Interactief Menu

Het programma start met een menu waarin je kunt kiezen:

```
======================================================================
AIRCRAFT LANDING SCHEDULING - SCENARIO MENU
======================================================================

Available Scenarios:

  1. Schiphol 1 Runway Light     (20 aircraft, ~10s)
  2. Schiphol 1 Runway Heavy     (35 aircraft, ~30s)
  3. Schiphol 2 Runways Medium   (30 aircraft, ~45s)
  4. Schiphol 3 Runways Heavy    (50 aircraft, ~1-3 min)
  5. Custom Sample Instance
  6. Load Custom File

Enter your choice (1-6): _
```

## 📁 Results Locatie

Alle resultaten worden opgeslagen in:
```
Operations_optimisation_Assignment_26/results/
├── scenario_1runway_light/
│   ├── schiphol_1runway_light_heuristic_gantt.png
│   ├── schiphol_1runway_light_optimal_gantt.png
│   ├── schiphol_1runway_light_comparison.png
│   ├── schiphol_1runway_light_cost_breakdown.png
│   └── tables/
│       ├── *_heuristic_detailed.csv
│       ├── *_heuristic_detailed.xlsx
│       ├── *_optimal_detailed.csv
│       └── *_optimal_detailed.xlsx
└── ...
```

## ✨ Features

- ✅ **Optimale MIP Solver**: Vindt de beste oplossing met PuLP
- ✅ **Greedy Heuristic**: Snelle benadering (< 1 seconde)
- ✅ **Multi-Runway Support**: 1 tot N runways
- ✅ **Visualisaties**: Gantt charts, cost breakdowns, vergelijkingen
- ✅ **Excel & CSV Export**: Gedetailleerde resultaten
- ✅ **Realistische Schiphol Scenarios**: Gebaseerd op echte wake turbulence data

## 📊 Wat krijg je?

Per scenario:
- **4 PNG visualisaties**:
  - Heuristic Gantt chart
  - Optimal Gantt chart
  - Side-by-side vergelijking
  - Cost breakdown per aircraft
- **2 CSV bestanden**: Gedetailleerde oplossingen
- **2 Excel bestanden**: Mooi geformatteerde oplossingen

## ⚡ Command-line opties

```bash
# Vanuit ROOT directory:
cd "/Users/jelleweijland/Documents/AE Vakken/Operations_Assignment/Operations_optimisation_Assignment_26"

# Run specifiek scenario
python aircraft_landing_scheduling/code/main.py --instance data/schiphol_1runway_light.txt --runways 1

# Custom tijd limit
python aircraft_landing_scheduling/code/main.py --instance sample --time-limit 600

# Batch experiment
python aircraft_landing_scheduling/code/main.py --batch --data data/ --runways 1

# Runway analysis
python aircraft_landing_scheduling/code/main.py --runway-analysis --instance sample --max-runways 4

# Help
python aircraft_landing_scheduling/code/main.py --help
```

## 📦 Dependencies

Zie `requirements.txt`:
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
pulp>=2.6.0
tabulate>=0.8.9
scipy>=1.7.0
seaborn>=0.11.0
openpyxl>=3.0.0
```

Installeer met:
```bash
source Operations_Guusje/bin/activate
pip install -r aircraft_landing_scheduling/requirements.txt
```

## 🏗️ Project Structuur

```
aircraft_landing_scheduling/
├── code/
│   ├── main.py                   # Entry point met interactief menu
│   ├── model.py                  # MIP optimalisatie model
│   ├── heuristic.py              # Greedy heuristic algoritme
│   ├── solver.py                 # Solver framework
│   ├── data_loader.py            # Data inlezen
│   ├── visualization.py          # Grafieken genereren
│   ├── utils.py                  # Helper functies
│   └── schiphol_scenarios.py     # Realistische scenarios
├── data/                         # Problem instances
├── requirements.txt              # Python dependencies
├── README.md                     # Deze file
└── INSTALL.md                    # Installatie instructies
```

## 🔬 Algoritmes

### 1. MIP Model (Optimaal)
- Mixed Integer Programming formulering
- Gebruikt PuLP met CBC solver
- Vindt gegarandeerd optimale oplossing
- Tijd: variabel (seconden tot minuten)

### 2. Greedy Heuristic (Snel)
- Constructieve greedy benadering
- Lokale verbetering fase
- Multi-start optie voor betere resultaten
- Tijd: < 1 seconde

## 📈 Code Begrijpelijkheid

De code is geschreven met **maximale leesbaarheid**:
- ✅ Uitgebreide Nederlandse comments
- ✅ Duidelijke variabele namen (geen cryptische afkortingen)
- ✅ Stap-voor-stap uitleg van complexe algoritmes
- ✅ Voorbeelden in de comments

Voorbeeld uit `model.py`:
```python
# CONSTRAINT 1: If ordering_variable = 1 (i lands before j)
# Then: landing_time_j ≥ landing_time_i + separation_i_before_j
# Using Big-M: landing_time_j ≥ landing_time_i + separation - Big_M * (1 - ordering_variable)
# When ordering_variable = 1: landing_time_j ≥ landing_time_i + separation (enforced!)
# When ordering_variable = 0: landing_time_j ≥ landing_time_i + separation - Big_M (inactive)
```

## 🐛 Troubleshooting

**"ModuleNotFoundError":**
```bash
source Operations_Guusje/bin/activate
pip install -r requirements.txt
```

**"Results folder op verkeerde plek":**
→ Zorg dat je runt vanuit de ROOT directory (zie boven)!

**"openpyxl not installed":**
```bash
pip install openpyxl
```

## 📚 Meer Info

Zie ook:
- `../README.md` - Root project documentatie
- `INSTALL.md` - Gedetailleerde installatie instructies
- `GET_STARTED.md` - Oorspronkelijke getting started guide

## 📖 Referenties

Beasley, J.E., Krishnamoorthy, M., Sharaiha, Y.M. and Abramson, D. (2000).
"Scheduling aircraft landings—the static case." Transportation Science, 34(2), 180-197.
