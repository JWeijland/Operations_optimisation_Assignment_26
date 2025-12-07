# 🚀 START HIER - Quick Reference

## Wat is dit project?

Aircraft Landing Scheduling optimalisatie voor Schiphol.
- **4 vaste scenarios** met bekende target times
- **Sensitivity analysis** voor verschillende combinaties
- **Twee oplossingsmethoden**: Greedy Heuristiek vs MILP Solver

---

## ⚡ Snel Starten

### Run de 4 Vaste Scenarios

```bash
# Interactief menu
python run_scenarios.py

# Specifiek scenario (1-4)
python run_scenarios.py --scenario 1

# Alle scenarios
python run_scenarios.py --all
```

**Output:** `results/scenario_X/` met Gantt charts, tabellen, en cost breakdowns

### Run Sensitivity Analysis

```bash
# Quick test (2x2, ~5 min)
python sensitivity_runner.py --quick-test

# Volledig (4x4, ~30 min)
python sensitivity_runner.py

# Custom
python sensitivity_runner.py --aircraft 20 30 40 --runways 1 2 3
```

**Output:** `results/[naam]/` met heatmaps en CSV data

---

## 📖 Documentatie

| Bestand | Inhoud |
|---------|--------|
| **README.md** | Volledige gebruikershandleiding |
| **PROJECT_STRUCTURE.md** | Bestandsstructuur en organisatie |
| **docs/HOE_WERKT_HET.md** | Simpele uitleg van de methoden |
| **docs/TECHNISCHE_UITLEG_MODEL.tex** | Wiskundige formulering (LaTeX) |

---

## 🎯 De 4 Scenarios

1. **Single Runway - Light** (20 aircraft, 1 runway)
2. **Single Runway - Heavy** (35 aircraft, 1 runway)
3. **Two Runways - Medium** (30 aircraft, 2 runways)
4. **Three Runways - Heavy** (50 aircraft, 3 runways)

Alle scenarios hebben **vaste, bekende target times** uit de data files.

---

## 📊 Output Begrijpen

### Gantt Chart
- **X-as**: Tijd (minuten vanaf 18:00)
- **Y-as**: Landingsbanen
- **Kleur**: Vroeg (blauw), On-time (groen), Laat (rood)

### Cost Heatmap (Sensitivity)
- **X-as**: Aantal runways
- **Y-as**: Aantal vliegtuigen
- **Kleur**: Kosten - Rood = hoog, Geel = laag

### Gap Heatmap
- **Waarde**: % verschil tussen heuristiek en optimaal
- **Typisch**: 5-15%

---

## 🔧 Meest Gebruikte Commando's

```bash
# Scenarios
python run_scenarios.py --all
python run_scenarios.py --scenario 2

# Sensitivity
python sensitivity_runner.py --quick-test
python sensitivity_runner.py

# Met custom time limit
python run_scenarios.py --scenario 1 --time-limit 100

# Rush hour vergelijking
python sensitivity_runner.py --rush-prob 0.3 --name low_rush
python sensitivity_runner.py --rush-prob 0.9 --name high_rush
```

---

## 📁 Belangrijke Folders

```
run_scenarios.py              ← Run de 4 scenarios
sensitivity_runner.py         ← Run sensitivity analysis

scenarios/                    ← Scenario definities
results/                      ← Output (wordt aangemaakt)
docs/                         ← Documentatie

aircraft_landing_scheduling/
├── code/                     ← Core optimalisatie code
└── data/                     ← Scenario data files
```

---

## 💡 Tips

1. **Start met**: `python run_scenarios.py --all`
   - Genereert alle basis resultaten
   - Duurt ~10-15 minuten
   - Output in `results/scenario_X/`

2. **Voor sensitivity**: `python sensitivity_runner.py --quick-test`
   - Snelle preview
   - Duurt ~5 minuten
   - Output in `results/quick_test/`

3. **Voor rapport**: Run beide bovenstaande
   - Gebruik Gantt charts van scenarios
   - Gebruik heatmaps van sensitivity
   - CSV files voor data tabellen

---

## ❓ Hulp Nodig?

- **Lijst scenarios**: `python run_scenarios.py --list`
- **Help**: `python run_scenarios.py --help`
- **Documentatie**: Lees `README.md`
- **Uitleg methoden**: Lees `docs/HOE_WERKT_HET.md`

---

## 🎓 Voor je Assignment

### Stap 1: Run Experimenten
```bash
python run_scenarios.py --all
python sensitivity_runner.py
```

### Stap 2: Analyseer Output
- Gantt charts in `results/scenario_X/figures/`
- Heatmaps in `results/[naam]/heatmaps/`
- Data in CSV files

### Stap 3: Schrijf Rapport
- Gebruik figures voor visualisaties
- Gebruik CSV data voor tabellen
- Verwijs naar `docs/TECHNISCHE_UITLEG_MODEL.tex` voor formules

---

**Succes! 🚀**
