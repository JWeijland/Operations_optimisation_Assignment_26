# 🎯 MAIN SCENARIO - Instructions

## Overzicht

Het **Main Scenario** voert 10 replications uit met **random target times** per run. Dit genereert robuuste statistieken voor je rapport.

### Belangrijkste kenmerken:
- ✅ **10 replications** met verschillende random target times
- ✅ **Consistente vliegtuig ratio** (CAT-B:10%, CAT-C:30%, CAT-D:45%, CAT-E:12%, CAT-F:3%)
- ✅ **Peak hour**: 11:00-11:40 (eerste 40 minuten)
- ✅ **Instelbaar peak percentage** (default 50%)
- ✅ **Time window**: earliest = target - 5 min, latest = target + 5 min (geclipped naar [0, 120])
- ✅ **Complete outputs** per replication: Gantt charts, Excel, comparison plots, cost breakdown

---

## 🚀 Gebruik

### Basis gebruik (default instellingen):
```bash
python run_main_scenario.py
```

Dit runt **10 replications** met:
- 49 vliegtuigen
- 3 runways
- 50% in peak hour (11:00-11:40)
- **Geen time limit** - solver runt tot 2% optimality gap bereikt is

### Met custom parameters:
```bash
# 10 replications, 50% peak
python run_main_scenario.py --replications 10 --peak-pct 0.50

# 20 replications, 70% peak
python run_main_scenario.py --replications 20 --peak-pct 0.70

# Custom aircraft en runways (ratio blijft hetzelfde!)
python run_main_scenario.py --aircraft 49 --runways 3
```

### Alle opties:
```
--replications INT    Aantal replications (default: 10)
--aircraft INT        Aantal vliegtuigen (default: 49, ratio blijft constant!)
--runways INT         Aantal runways (default: 3)
--peak-pct FLOAT      Peak percentage (default: 0.50 = 50%)
--seed INT            Base random seed (default: 1000)

SOLVER SETTINGS (niet instelbaar, vast ingesteld):
  - Geen time limit
  - 2% optimality gap
  - CBC solver
```

---

## 📁 Output Structuur

Na het runnen krijg je:

```
results/main_scenario/
├── replication_1/
│   ├── rep1_heuristic_gantt.png          # Gantt chart heuristic
│   ├── rep1_optimal_gantt.png            # Gantt chart optimal
│   ├── rep1_comparison.png               # Comparison plot
│   ├── rep1_cost_breakdown.png           # Cost breakdown
│   └── tables/
│       ├── rep1_heuristic_detailed.xlsx  # ⭐ Excel zoals schiphol_3runways_heavy_optimal_detailed.xlsx
│       └── rep1_optimal_detailed.xlsx    # ⭐ Excel zoals schiphol_3runways_heavy_optimal_detailed.xlsx
│
├── replication_2/
│   └── ... (zelfde structuur)
│
├── replication_3/
│   └── ...
│
├── ...
│
├── replication_10/
│   └── ...
│
├── replication_results.csv              # Alle results per replication
└── statistics_summary.txt               # Statistieken (mean, std, min, max)
```

---

## 📊 Output per Replication

Elke replication krijgt **exact** wat je nodig hebt:

### 1. Gantt Charts
- `rep{N}_heuristic_gantt.png` - Heuristic solution schedule
- `rep{N}_optimal_gantt.png` - Optimal solution schedule

### 2. Comparison & Breakdown
- `rep{N}_comparison.png` - Vergelijking heuristic vs optimal
- `rep{N}_cost_breakdown.png` - Cost breakdown per aircraft

### 3. Excel Tables (ZOALS JE WILDE!)
- `rep{N}_heuristic_detailed.xlsx` - Compleet zoals `schiphol_3runways_heavy_optimal_detailed.xlsx`
  - Header met metadata (cost, time, status)
  - Tabel met: Aircraft ID, Runway, Earliest Time, Target Time, Latest Time, Actual Landing, Deviation, Status, Penalties, Cost
  - Color coding (Early=geel, Late=rood, On-time=groen)
  - TOTAL row onderaan

- `rep{N}_optimal_detailed.xlsx` - Zelfde voor optimal

---

## 📈 Statistieken

Na alle replications krijg je:

### `replication_results.csv`:
```
replication,seed,heuristic_cost,optimal_cost,gap_percent,heuristic_time,optimal_time,optimal_status
1,1001,65.95,19.78,233.33,0.004,30.117,Optimal
2,1002,188.64,108.34,74.11,0.005,30.421,Optimal
...
```

### `statistics_summary.txt`:
```
Metric               Mean            Std             Min             Max
--------------------------------------------------------------------------------
Heuristic Cost       127.30          86.76           65.95           188.64
Optimal Cost         64.06           62.62           19.78           108.34
Gap (%)              153.72          112.59          74.11           233.33
```

**Deze statistieken zijn perfect voor je rapport!**

---

## 🔧 Technische Details

### Random Target Times
Per replication worden **nieuwe random target times** gegenereerd:
- **50%** (instelbaar) van vliegtuigen krijgt target in peak hour (0-40 min)
- **50%** verdeeld over off-peak (40-120 min)
- Earliest = max(0, target - 5)
- Latest = min(120, target + 5)

### Vaste Elementen (consistent over alle replications)
- ✅ Vliegtuig categorie ratio: CAT-B:10%, CAT-C:30%, CAT-D:45%, CAT-E:12%, CAT-F:3%
- ✅ Aircraft types: B747, B777, B787, A330, B767, A300, B737, A320, E190, E175, ATR-72, Citations
- ✅ Separation matrix: ICAO wake turbulence (60-140 seconden)
- ✅ Cost structure: Per categorie (CAT-B duurste, CAT-F goedkoopste)

### Variabele Elementen (verschillend per replication)
- ❌ Target times (random!)
- ❌ Aircraft volgorde (random shuffle)

---

## ⚙️ Hoe het werkt

```python
# Voor elke replication:
1. Genereer 49 vliegtuigen met vaste ratio
2. Assign random target time (50% in peak, 50% off-peak)
3. Bereken earliest/latest (target ± 5 min, clipped)
4. Create separation matrix (ICAO)
5. Solve heuristic
6. Solve optimal (time limit 50s)
7. Generate alle outputs (Gantt, Excel, plots)
8. Save results

# Na alle replications:
9. Bereken statistieken (mean, std, min, max)
10. Save summary
```

---

## 🎯 Voor je Rapport

### Wat te gebruiken:

**Individuele Replications:**
- Kies 1-3 "representatieve" replications (bijv. rep met mean cost)
- Gebruik Excel tables: `rep{N}_optimal_detailed.xlsx`
- Gebruik Gantt charts: `rep{N}_optimal_gantt.png`

**Statistieken:**
- Gebruik `statistics_summary.txt` voor gemiddelde results
- Citeer mean ± std voor robuustheid
- Toon range (min-max) voor variabiliteit

**Voorbeeld rapportage:**
> "Het Main Scenario is 10 keer gerund met random target times (50% in peak hour 11:00-11:40).
> De gemiddelde optimal cost was €64.06 ± €62.62, met een range van €19.78 tot €108.34.
> De heuristic behaalde €127.30 ± €86.76, resulterend in een gemiddelde gap van 153.72%."

---

## ✅ Verificatie Checklist

Na runnen, check:
- [ ] 10 replication folders aanwezig
- [ ] Elk heeft 4 PNG files (2 Gantt, 1 comparison, 1 breakdown)
- [ ] Elk heeft 2 Excel files (heuristic & optimal detailed)
- [ ] Excel files hebben zelfde format als `schiphol_3runways_heavy_optimal_detailed.xlsx`
- [ ] `replication_results.csv` heeft 10 rows
- [ ] `statistics_summary.txt` toont mean, std, min, max
- [ ] Alle costs zijn positief en logisch

---

## 🔄 Als je opnieuw wilt runnen

```bash
# Verwijder oude results
rm -rf results/main_scenario

# Run opnieuw
python run_main_scenario.py
```

---

## 💡 Tips

1. **Voor snelle test**: Gebruik `--replications 2` eerst
2. **Voor diepere analyse**: Gebruik `--replications 20`
3. **Voor verschillende scenarios**: Verander `--peak-pct` (bijv. 0.30, 0.70)
4. **Check altijd**: `statistics_summary.txt` voor overall beeld

---

## ❓ Vragen?

Als iets niet werkt of je wilt iets aanpassen, check:
- `run_main_scenario.py` - Main script
- `aircraft_landing_scheduling/code/solver.py` - Solver logic
- `aircraft_landing_scheduling/code/visualization.py` - Plotting
- `aircraft_landing_scheduling/code/utils.py` - Excel export

**Alles is ready voor je rapport!** 🎓✨
