# Aircraft Landing Scheduling - Optimalisatie Project

Dit project optimaliseert landingsschema's voor vliegtuigen op Schiphol met als doel de totale kosten te minimaliseren.

## 📁 Project Structuur

```
.
├── run_scenarios.py              # Run de 4 vaste scenarios
├── sensitivity_runner.py         # Run sensitivity analysis
├── scenarios/
│   └── scenario_definitions.py   # Definitie van de 4 scenarios
├── aircraft_landing_scheduling/
│   ├── code/                     # Core optimalisatie code
│   │   ├── solver.py            # Heuristiek + MILP solver
│   │   ├── model.py             # MILP formulering
│   │   ├── heuristic.py         # Greedy heuristiek
│   │   ├── data_loader.py       # Data inlezen
│   │   └── visualization.py     # Gantt charts en plots
│   └── data/                     # Scenario data files
│       ├── schiphol_1runway_light.txt
│       ├── schiphol_1runway_heavy.txt
│       ├── schiphol_2runways_medium.txt
│       └── schiphol_3runways_heavy.txt
├── results/                      # Output (wordt aangemaakt)
└── docs/
    ├── HOE_WERKT_HET.md         # Uitleg hoe de methoden werken
    └── TECHNISCHE_UITLEG_MODEL.tex  # Wiskundige formulering
```

## 🚀 Quick Start

### 1. Run de 4 Vaste Scenarios

**Interactief menu:**
```bash
python run_scenarios.py
```

**Specifiek scenario:**
```bash
python run_scenarios.py --scenario 1   # Single runway, light traffic
python run_scenarios.py --scenario 2   # Single runway, heavy traffic
python run_scenarios.py --scenario 3   # Two runways, medium traffic
python run_scenarios.py --scenario 4   # Three runways, heavy traffic
```

**Alle scenarios:**
```bash
python run_scenarios.py --all
```

**Lijst scenarios:**
```bash
python run_scenarios.py --list
```

### 2. Run Sensitivity Analysis

Test verschillende combinaties van vliegtuigen en runways:

```bash
# Quick test (20, 30 aircraft × 1, 2 runways)
python sensitivity_runner.py --quick-test

# Volledig (20, 30, 40, 50 aircraft × 1, 2, 3, 4 runways)
python sensitivity_runner.py

# Custom
python sensitivity_runner.py \
    --aircraft 20 30 40 \
    --runways 1 2 3 \
    --rush-prob 0.7 \
    --name my_analysis
```

## 📊 De 4 Scenarios

### Scenario 1: Single Runway - Light Traffic
- **Aircraft**: 20
- **Runways**: 1
- **Traffic**: Off-peak
- **Gebruik**: Toon baseline performance

### Scenario 2: Single Runway - Heavy Traffic
- **Aircraft**: 35
- **Runways**: 1
- **Traffic**: Peak hour
- **Gebruik**: Toon impact van congestie op 1 runway

### Scenario 3: Two Runways - Medium Traffic
- **Aircraft**: 30
- **Runways**: 2
- **Traffic**: Medium load
- **Gebruik**: Toon voordeel van 2de runway

### Scenario 4: Three Runways - Heavy Traffic
- **Aircraft**: 50
- **Runways**: 3
- **Traffic**: Full rush hour
- **Gebruik**: Toon schaalbaarheid met 3 runways

## 📈 Output

Na het runnen van scenarios vind je:

### Per Scenario:
```
results/scenario_X/
├── figures/
│   ├── scenario_X_heuristic_gantt.png     # Gantt chart heuristiek
│   ├── scenario_X_optimal_gantt.png       # Gantt chart optimaal
│   ├── scenario_X_comparison.png          # Vergelijking beide methoden
│   └── scenario_X_cost_breakdown.png      # Kostenverdeling
└── tables/
    ├── scenario_X_heuristic.csv           # Heuristiek oplossing
    ├── scenario_X_optimal.csv             # Optimale oplossing
    ├── scenario_X_heuristic.xlsx          # Excel versie
    └── scenario_X_optimal.xlsx
```

### Sensitivity Analysis:
```
results/[experiment_name]/
├── sensitivity_results_YYYYMMDD_HHMMSS.csv   # Alle resultaten
└── heatmaps/
    ├── optimal_cost_heatmap.png              # Hoofdresultaat!
    └── gap_percent_heatmap.png               # Kwaliteit heuristiek
```

## 🔧 Hoe het werkt

Het project gebruikt **twee oplossingsmethoden**:

### 1. Greedy Heuristiek (Snel)
- Plant vliegtuigen één voor één
- Kiest steeds beste optie voor huidig vliegtuig
- **Tijd**: 0.01 - 0.1 seconden
- **Kwaliteit**: 5-15% slechter dan optimaal

### 2. MILP Solver (Optimaal)
- Zoekt globaal beste oplossing
- Gebruikt branch-and-bound algoritme
- **Tijd**: 10 - 50 seconden
- **Kwaliteit**: Optimaal (binnen time limit)

Zie `docs/HOE_WERKT_HET.md` voor uitgebreide uitleg.

## 📊 Sensitivity Analysis

De sensitivity analysis test systematisch verschillende combinaties:

**Parameters:**
- `--aircraft`: Lijst van aantallen vliegtuigen (bijv. `20 30 40 50`)
- `--runways`: Lijst van aantallen runways (bijv. `1 2 3 4`)
- `--rush-prob`: Percentage in rush hour (0.0 - 1.0)
- `--time-limit`: Tijd per scenario in seconden
- `--name`: Experiment naam

**Voorbeeld:**
```bash
# Test impact rush hour
python sensitivity_runner.py --rush-prob 0.3 --name low_rush
python sensitivity_runner.py --rush-prob 0.7 --name high_rush

# Vergelijk de heatmaps!
```

## 📝 Resultaten Interpreteren

### Gantt Chart
- **X-as**: Tijd (minuten vanaf 18:00)
- **Y-as**: Landingsbanen
- **Blokken**: Vliegtuigen met landingstijd
- **Kleur**: Vroeg (blauw), On-time (groen), Laat (rood)

### Cost Heatmap
- **X-as**: Aantal runways
- **Y-as**: Aantal vliegtuigen
- **Kleur**: Kosten (EUR) - Rood = hoog, Geel = laag
- **Trend**: Kosten dalen bij meer runways

### Gap Heatmap
- **Waarde**: (Heuristiek - Optimaal) / Optimaal × 100%
- **Interpretatie**: Hoe goed is de heuristiek?
- **Typisch**: 5-15% gap

## 🎯 Veelgebruikte Commando's

```bash
# Scenario's runnen
python run_scenarios.py --scenario 1
python run_scenarios.py --all

# Sensitivity analysis
python sensitivity_runner.py --quick-test          # Quick (5 min)
python sensitivity_runner.py                       # Full (30-60 min)

# Custom time limit
python run_scenarios.py --scenario 2 --time-limit 100
python sensitivity_runner.py --time-limit 100

# Rush hour vergelijking
python sensitivity_runner.py --rush-prob 0.3 --name low
python sensitivity_runner.py --rush-prob 0.9 --name high
```

## 📖 Documentatie

- **HOE_WERKT_HET.md**: Simpele uitleg van de twee methoden
- **TECHNISCHE_UITLEG_MODEL.tex**: Wiskundige formulering (LaTeX)
- Deze README: Gebruikershandleiding

## ⚙️ Technische Details

### Solver Settings
- **Time limit**: 50 seconden (default)
- **Optimality gap**: 1%
- **Solver**: PuLP met CBC

### Aircraft Types
- **Heavy** (30%): B747, B777, A330 - Separatie 90-180s
- **Medium** (60%): B737, A320 - Separatie 60-120s
- **Light** (10%): Citation, Phenom - Separatie 60s

### Cost Structure
- **Heavy**: €100/min vroeg, €200/min laat
- **Medium**: €60/min vroeg, €150/min laat
- **Light**: €30/min vroeg, €80/min laat

## 🎓 Voor je Rapport

**Aanbevolen analyses:**

1. **Run alle 4 scenarios**
   ```bash
   python run_scenarios.py --all
   ```
   → Gebruik Gantt charts en cost breakdown

2. **Rush hour impact**
   ```bash
   python sensitivity_runner.py --rush-prob 0.3 --name low_rush
   python sensitivity_runner.py --rush-prob 0.7 --name high_rush
   ```
   → Vergelijk heatmaps side-by-side

3. **Runway scaling**
   ```bash
   python sensitivity_runner.py
   ```
   → Analyseer diminishing returns van extra runways

**Key metrics om te rapporteren:**
- Optimal cost per scenario
- Gap percentage (heuristiek vs optimaal)
- Solve tijd (schaalbaarheid)
- Cost reduction door extra runways
- Rush hour impact op kosten

---

**Gemaakt voor Operations Optimisation Assignment 2024**
