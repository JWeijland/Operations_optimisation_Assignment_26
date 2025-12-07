# Project Structuur - Overzicht

## 📂 Hoofdbestanden

### Scripts om te runnen:

```
run_scenarios.py          → Run de 4 vaste scenarios
sensitivity_runner.py     → Run sensitivity analysis
```

### Configuratie:

```
scenarios/
└── scenario_definitions.py   → Definitie van de 4 scenarios
```

## 📂 Core Code

```
aircraft_landing_scheduling/
├── code/
│   ├── solver.py              → Main solver (heuristiek + MILP)
│   ├── model.py               → MILP formulering (wiskundige model)
│   ├── heuristic.py           → Greedy heuristiek algoritme
│   ├── data_loader.py         → Data inlezen en valideren
│   ├── visualization.py       → Gantt charts en plots
│   ├── utils.py               → Export functies (CSV, Excel)
│   ├── schiphol_scenarios.py  → Scenario generator (voor sensitivity)
│   ├── sensitivity_config.py  → Sensitivity analysis configuratie
│   ├── sensitivity_analysis.py → Sensitivity analysis engine
│   └── sensitivity_visualization.py → Heatmaps
│
└── data/
    ├── schiphol_1runway_light.txt       → Scenario 1 data
    ├── schiphol_1runway_heavy.txt       → Scenario 2 data
    ├── schiphol_2runways_medium.txt     → Scenario 3 data
    ├── schiphol_3runways_heavy.txt      → Scenario 4 data
    └── sensitivity_*.txt                 → Gegenereerde sensitivity data
```

## 📂 Documentatie

```
docs/
├── HOE_WERKT_HET.md              → Simpele uitleg hoe methoden werken
├── TECHNISCHE_UITLEG_MODEL.tex   → Wiskundige formulering (LaTeX)
└── README_OLD.md                  → Oude README (backup)

README.md                          → Hoofddocumentatie (start hier!)
PROJECT_STRUCTURE.md               → Dit bestand
```

## 📂 Output (wordt gegenereerd)

### Na run_scenarios.py:

```
results/
├── scenario_1/
│   ├── figures/
│   │   ├── scenario_1_heuristic_gantt.png
│   │   ├── scenario_1_optimal_gantt.png
│   │   ├── scenario_1_comparison.png
│   │   └── scenario_1_cost_breakdown.png
│   └── tables/
│       ├── scenario_1_heuristic.csv
│       ├── scenario_1_optimal.csv
│       ├── scenario_1_heuristic.xlsx
│       └── scenario_1_optimal.xlsx
│
├── scenario_2/
├── scenario_3/
└── scenario_4/
```

### Na sensitivity_runner.py:

```
results/
└── [experiment_name]/
    ├── sensitivity_results_YYYYMMDD_HHMMSS.csv
    └── heatmaps/
        ├── optimal_cost_heatmap.png
        ├── gap_percent_heatmap.png
        ├── solve_time_heatmap.png
        └── speedup_heatmap.png
```

## 📂 Configuraties (sensitivity analysis)

```
configs/
├── default_sensitivity.yaml
├── extreme_rush_sensitivity.yaml
├── no_rush_sensitivity.yaml
└── heavy_traffic_sensitivity.yaml
```

## 🗑️ Wat is verwijderd/genegeerd:

- `Operations_Guusje/` - Python virtual environment
- `.git/` - Git repository data
- `__pycache__/` - Python cache bestanden
- Oude test en sample scripts
- Duplicate documentatie files

## 🎯 Voor je Assignment

### Gebruik deze files:

**Voor rapportage:**
1. `README.md` - Projectbeschrijving
2. `docs/HOE_WERKT_HET.md` - Uitleg methoden
3. `docs/TECHNISCHE_UITLEG_MODEL.tex` - Wiskundige formulering

**Voor experimenten:**
1. `run_scenarios.py --all` - Run alle 4 scenarios
2. `sensitivity_runner.py` - Run sensitivity analysis

**Voor output:**
1. `results/scenario_X/` - Per scenario resultaten
2. `results/[experiment]/` - Sensitivity analysis resultaten

### Negeer deze files:

- `Operations_Guusje/` - Virtual environment
- `.git/` - Versie controle
- `__pycache__/` - Python cache
- `*.pyc` - Compiled Python files

## 📋 Snelle Referentie

| Wat wil je? | Run dit: |
|-------------|----------|
| Scenario 1 runnen | `python run_scenarios.py --scenario 1` |
| Alle scenarios runnen | `python run_scenarios.py --all` |
| Sensitivity analysis (quick) | `python sensitivity_runner.py --quick-test` |
| Sensitivity analysis (full) | `python sensitivity_runner.py` |
| Lijst scenarios | `python run_scenarios.py --list` |
| Help | `python run_scenarios.py --help` |

## 🔧 Code Organisatie

### Layer 1: Data (data_loader.py)
- Leest scenario files
- Valideert data
- Maakt ProblemInstance objecten

### Layer 2: Oplossers
- **heuristic.py** - Greedy algoritme
- **model.py** - MILP formulering
- **solver.py** - Combineert beide + vergelijkt

### Layer 3: Visualisatie (visualization.py)
- Gantt charts
- Comparison plots
- Cost breakdowns

### Layer 4: Analysis
- **sensitivity_analysis.py** - Run meerdere scenarios
- **sensitivity_visualization.py** - Heatmaps

### Layer 5: Main Scripts
- **run_scenarios.py** - Voor vaste scenarios
- **sensitivity_runner.py** - Voor sensitivity analysis
