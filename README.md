# Aircraft Landing Scheduling Optimization

Mixed Integer Programming (MIP) implementatie voor het Aircraft Landing Scheduling probleem, gebaseerd op Beasley et al. (2000).

## 📁 Project Structuur

```
Operations_optimisation_Assignment_26/
├── aircraft_landing_scheduling/
│   └── code/                      # Core Python modules
│       ├── data_loader.py         # Data loading en parsing
│       ├── model.py               # MIP model (Beasley et al. 2000)
│       ├── heuristic.py           # Greedy heuristic
│       ├── solver.py              # High-level solver interface
│       ├── schiphol_scenarios.py  # Scenario generator
│       ├── sensitivity_config.py  # Sensitivity analysis configuratie
│       ├── sensitivity_analysis.py # Sensitivity analysis module
│       ├── utils.py               # Hulpfuncties
│       └── visualization.py       # Gantt charts en visualisaties
│
├── data/                          # Scenario bestanden (3 files)
│   ├── schiphol_1runway_heavy.txt
│   ├── schiphol_1runway_light.txt
│   └── schiphol_3runways_heavy.txt
│
├── results/                       # Resultaten
│   ├── scenario_1runway_heavy/   # Scenario 1 resultaten
│   ├── scenario_3runways_heavy/  # Scenario 2 resultaten
│   └── sensitivity_analysis/     # Sensitivity analysis resultaten
│       ├── heatmaps/             # Heatmap visualisaties
│       └── sensitivity_results_20251207_152344.csv  # 80 scenarios × 5 reps
│
├── support_documents/             # Papers en assignment docs
│
├── run_scenarios.py               # ⭐ Runner voor 4 scenarios
├── sensitivity_runner.py          # ⭐ Runner voor sensitivity analysis
├── requirements.txt               # Python dependencies
├── optimalisatie_methode.tex      # LaTeX documentatie van methode
└── README.md                      # Deze file
```

## 🚀 Quick Start

### Installatie

```bash
# Maak virtual environment (optioneel maar aanbevolen)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installeer dependencies
pip install -r requirements.txt
```

### 1. Run de 4 Scenarios

```bash
python run_scenarios.py
```

Dit runt automatisch:
- ✅ Scenario 1: 1 runway, heavy traffic (gebruikt bestaand data file)
- ✅ Scenario 2: 1 runway, light traffic (gebruikt bestaand data file)
- ✅ Scenario 3: 3 runways, heavy traffic (gebruikt bestaand data file)
- ✅ Scenario 4: 3 runways, light traffic (gegenereerd on-the-fly)

**Output per scenario:**
- `results/scenario_*/comparison_results.csv` - Heuristic vs Optimal
- `results/scenario_*/gantt_*.png` - Gantt charts
- `results/scenario_*/*_solution.csv` - Detailed schedules

### 2. Run Sensitivity Analysis

```bash
python sensitivity_runner.py
```

**Configuratie** (`aircraft_landing_scheduling/code/sensitivity_config.py:76`):
- Aircraft counts: `[20, 30, 40, 50]`
- Runway counts: `[1, 2, 3, 4]`
- **Replications: 5 per scenario**
- Time limit: 50s per instance
- **Totaal: 80 scenarios** (4×4×5)
- **Tijd: ~60-90 minuten**

**Output:**
- `results/sensitivity_analysis/sensitivity_results_TIMESTAMP.csv` (80 rows)
- `results/sensitivity_analysis/heatmaps/`:
  - `optimal_cost_heatmap.png` - Gemiddelde kosten
  - `gap_percent_heatmap.png` - Heuristic performance

## 🧮 Optimalisatie Methode

### Model Type
**Mixed Integer Linear Programming (MILP)** volgens Beasley et al. (2000):
> "Scheduling Aircraft Landings—The Static Case"
> Transportation Science, 34(2), 180-197.

### Objective Function
```
minimize: Σ (g_i · α_i + h_i · β_i)
```
- `g_i`: Penalty vroeg landen (€/min)
- `h_i`: Penalty laat landen (€/min)
- `α_i`: Minuten vroeg
- `β_i`: Minuten laat

### Decision Variables
- `x_i ∈ ℝ⁺`: Landing tijd vliegtuig i
- `α_i, β_i ∈ ℝ⁺`: Afwijking van target tijd
- `δ_ij ∈ {0,1}`: Binary ordering (i landt voor j)
- `y_ir ∈ {0,1}`: Runway assignment (i op runway r)

### Key Constraints
1. **Time windows**: `E_i ≤ x_i ≤ L_i`
2. **Target deviation**: `x_i = T_i - α_i + β_i`
3. **Separation** (Big-M methode):
   - Zelfde runway: minimaal `S_ij` seconden
   - Andere runways: minimaal `s_ij` seconden
4. **Runway assignment**: Elk vliegtuig → 1 runway

Zie `optimalisatie_methode.tex` voor volledige wiskundige formulering.

### Solver
**Default:** CBC (COIN-OR Branch and Cut) via PuLP
- ✅ Gratis en open-source
- ✅ Geen licentie nodig
- ⚡ Alternatief: Gurobi (als licentie beschikbaar)

**Settings:**
- Time limit: 50s per instance
- MIP gap: 1%
- Branch-and-Bound algoritme

## 📊 Resultaten Interpretatie

### Heuristic Gap
```
Gap = (Heuristic Cost - Optimal Cost) / Optimal Cost × 100%
```

**Interpretatie:**
- `Gap = 0%`: Heuristic vond optimale oplossing 🎯
- `Gap = 10%`: Heuristic is 10% duurder (redelijk)
- `Gap < 0%`: **Timeout** - solver vond geen echte optimale oplossing

### Negatieve Gaps (< 0%)
⚠️ In sensitivity analysis komen negatieve gaps voor bij:
- **40-50 vliegtuigen met 3-4 runways**
- Time limit van 50s is te kort voor deze complexe instances
- Heuristic heeft betere oplossing dan solver binnen time limit

**Dit is OK!** Het toont:
- Computational complexity van grote instances
- Trade-off tussen solution quality en solve tijd
- Waarde van goede heuristics voor real-time beslissingen

**In rapport schrijven:**
> "Voor scenarios met 40+ vliegtuigen en 3-4 runways was de 50-seconden time limit onvoldoende. In 15 van 80 scenarios (18.8%) bereikte de solver de time limit, wat resulteerde in mogelijk suboptimale oplossingen. Dit toont de computational complexity van grote-schaal aircraft landing scheduling problemen."

## 🎯 Key Features

✅ Complete MIP formulering volgens Beasley et al. (2000)
✅ Greedy heuristic voor snelle oplossingen (< 0.01s)
✅ Multiple runway support (1-4 runways)
✅ Realistic Schiphol evening rush scenarios
✅ Sensitivity analysis met 5 replications
✅ Automatische Gantt chart visualisaties
✅ Heatmap generatie (gemiddelde over replications)
✅ CBC solver (gratis) + Gurobi support

## 📈 Sensitivity Analysis Details

### Scenario Generatie
Elke scenario krijgt unique random seed:
```python
seed = base_seed + (aircraft × 1000) + (runways × 100) + replication
```

Dit zorgt voor:
- **Reproduceerbare** resultaten (vaste base_seed = 42)
- **Verschillende** rush hour patronen per replication
- **Consistente** vergelijking tussen scenarios

### Aggregatie over Replications
Heatmaps tonen **gemiddelde** van 5 replications:

```python
df_agg = df.groupby(['num_aircraft', 'num_runways']).agg({
    'optimal_cost': 'mean',
    'gap_percent': 'mean',
    'optimal_time_s': 'mean'
})
```

Elke cel in heatmap = gemiddelde van 5 runs → **robuustere** resultaten!

## 📖 Bestandsoverzicht

### Essentiële Bestanden

| Bestand | Functie |
|---------|---------|
| `run_scenarios.py` | Runner voor 4 main scenarios |
| `sensitivity_runner.py` | Runner voor sensitivity analysis |
| `optimalisatie_methode.tex` | LaTeX documentatie methode |
| `requirements.txt` | Python dependencies |

### Core Code Modules

| Module | Beschrijving |
|--------|--------------|
| `model.py` | MIP model volgens Beasley (2000) |
| `solver.py` | High-level interface (heuristic + optimal) |
| `heuristic.py` | Greedy + multi-start heuristics |
| `schiphol_scenarios.py` | Realistische scenario generator |
| `sensitivity_config.py` | Config voor sensitivity analysis |
| `data_loader.py` | Parse data files |
| `visualization.py` | Gantt charts maken |
| `utils.py` | Helper functies |

## 🔧 Aanpassingen Maken

### Sensitivity Analysis Parameters Wijzigen

Edit `aircraft_landing_scheduling/code/sensitivity_config.py`:

```python
# Lijn 56-59: Scenario parameters
aircraft_counts: List[int] = [20, 30, 40, 50]  # Wijzig hier
runway_counts: List[int] = [1, 2, 3, 4]        # Wijzig hier

# Lijn 72: Solver time limit
time_limit: int = 50  # Verhoog voor complexe scenarios

# Lijn 76: Aantal replications
num_replications: int = 5  # Wijzig voor meer/minder herhalingen
```

### Andere Rush Hour Probability

```python
# Lijn 23: Rush hour probability
probability: float = 0.5  # 0.0-1.0, waar 0.5 = 50%
```

## 🎓 Voor je Rapport

### Aanbevolen Analyses

1. **Run alle 4 scenarios**
   ```bash
   python run_scenarios.py
   ```
   → Gebruik Gantt charts voor visuele vergelijking

2. **Volledige sensitivity analysis**
   ```bash
   python sensitivity_runner.py
   ```
   → Analyseer heatmaps voor trends

### Key Metrics om te Rapporteren

- ✅ Optimal cost per scenario
- ✅ Heuristic gap percentage
- ✅ Solve tijd (schaalbaarheid)
- ✅ Cost reduction door extra runways
- ✅ Impact van rush hour density
- ✅ Percentage timeouts (negatieve gaps)

### Tabellen in CSV

Alle resultaten zijn opgeslagen als CSV:
- `results/scenario_*/comparison_results.csv`
- `results/sensitivity_analysis/sensitivity_results_*.csv`

Direct te importeren in Excel/LaTeX voor je rapport!

## 📚 Referenties

Beasley, J. E., Krishnamoorthy, M., Sharaiha, Y. M., & Abramson, D. (2000).
*Scheduling aircraft landings—The static case.*
Transportation Science, 34(2), 180-197.

---

**AE4441-16 Operations Optimisation**
TU Delft, 2024-2025
