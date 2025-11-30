# Quick Start Guide

## Installation (5 minutes)

```bash
cd aircraft_landing_scheduling
pip install -r requirements.txt
```

## Run Your First Experiment (2 minutes)

```bash
python code/main.py --instance sample --runways 1
```

## Understanding Output

You'll see:
1. **Heuristic Solution** - Fast, near-optimal
2. **Optimal Solution** - Proven best solution
3. **Comparison** - Quality gap and speedup
4. **Visualizations** - Saved to `results/figures/`

## Next Steps

### Run Batch Experiments
```bash
python code/main.py --batch --data data/ --runways 1
```

### Analyze Multiple Runways
```bash
python code/main.py --runway-analysis --instance sample --max-runways 4
```

### View Results
- Tables: `results/tables/`
- Figures: `results/figures/`

## Project Structure

```
aircraft_landing_scheduling/
├── code/              # Source code
├── data/              # Test instances
├── results/           # Output files
├── report/            # LaTeX report
└── tests/             # Unit tests
```

Happy scheduling! ✈️
