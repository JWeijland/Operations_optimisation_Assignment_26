# Complete Sensitivity Analysis - Instructions

## Quick Start (One Button Solution!)

To run the complete sensitivity analysis for your report, simply run:

```bash
cd aircraft_landing_scheduling/code
python run_complete_sensitivity.py
```

That's it! Now go get a coffee (or several) while it runs...

## What Will Happen?

The script will:

1. **Run 15 scenarios** (5 aircraft counts × 3 runway counts):
   - Aircraft: 10, 20, 30, 40, 50
   - Runways: 1, 2, 3
   - Time limit: 300 seconds per scenario
   - **Aircraft mix**: 30% Heavy, 67% Medium, 3% Light (target ratios)
     - This reflects typical European airport operations
     - **Important**: Each instance has at least 1 light aircraft (operational reality)
     - Note: Ratios are rounded to nearest integer, adjusted to ensure ≥1 light
     - Example: 10 aircraft = 3H, 6M, 1L (30/60/10)
     - Example: 20 aircraft = 6H, 13M, 1L (30/65/5)
     - Example: 30 aircraft = 9H, 20M, 1L (30/67/3 - matches target!)
     - Example: 50 aircraft = 15H, 34M, 1L (30/68/2)

2. **Generate outputs**:
   - **Comprehensive results table** (CSV + Excel) with ALL details:
     - Number of variables
     - Number of constraints
     - Solve time
     - Optimality gap
     - Cost (optimal and heuristic)
     - Status
   - **Heatmaps** showing:
     - Optimal cost by aircraft/runway count
     - Solve time by aircraft/runway count
     - Gap percentage by aircraft/runway count
     - Cost reduction compared to single runway
     - Speedup of heuristic vs optimal
   - **Comparison plots** showing trends

3. **Save everything** to: `results/complete_sensitivity_analysis/`

## Expected Runtime

- **Minimum**: ~15 minutes (if all scenarios solve quickly)
- **Maximum**: ~75 minutes (if many hit the 300s time limit)
- **Typical**: 30-45 minutes

The script will show progress as it runs, so you can see how much longer it will take.

## Output Files

After completion, you'll find:

```
results/complete_sensitivity_analysis/
├── comprehensive_results_table.csv      ← THE BIG TABLE (all runs)
├── comprehensive_results_table.xlsx     ← Same but in Excel
├── sensitivity_results_TIMESTAMP.csv    ← Raw results
├── heatmaps/
│   ├── optimal_cost_heatmap.png
│   ├── solve_time_heatmap.png
│   ├── gap_percent_heatmap.png
│   ├── cost_reduction_heatmap.png
│   └── speedup_heatmap.png
└── plots/
    └── comparison_plots.png
```

## The Comprehensive Results Table

The main table (`comprehensive_results_table.csv`) contains these columns:

| Column | Description |
|--------|-------------|
| `num_aircraft` | Number of aircraft (10, 20, 30, 40, 50) |
| `num_runways` | Number of runways (1, 2, 3) |
| `replication` | Replication number (always 0 for this analysis) |
| `num_variables` | Number of decision variables in the MILP |
| `num_constraints` | Number of constraints in the MILP |
| `optimal_cost` | Total cost from MILP solver |
| `heuristic_cost` | Total cost from FCFS heuristic |
| `gap_percent` | Gap between heuristic and optimal (%) |
| `optimal_time_s` | Time to solve MILP (seconds) |
| `heuristic_time_s` | Time to run heuristic (seconds) |
| `speedup` | How much faster the heuristic is |
| `scenario_name` | Unique scenario identifier |
| `seed` | Random seed used |
| `status` | Completion status |

## Using Results in Your Report

### For LaTeX Tables

The CSV file can be directly imported into LaTeX using `pgfplotstable` or converted to LaTeX format:

```python
import pandas as pd
df = pd.read_csv('comprehensive_results_table.csv')
print(df.to_latex(index=False))
```

### For Figures

All heatmaps and plots are saved as high-resolution PNG files (300 DPI) ready for inclusion in your report:

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{../results/complete_sensitivity_analysis/heatmaps/optimal_cost_heatmap.png}
\caption{Optimal landing cost by aircraft count and runway count}
\label{fig:cost_heatmap}
\end{figure}
```

## Customization

If you want to change the parameters, edit `run_complete_sensitivity.py`:

```python
config = SensitivityAnalysisConfig(
    experiment_name="complete_sensitivity_10_to_50_aircraft",
    aircraft_counts=[10, 20, 30, 40, 50],  # ← Change these
    runway_counts=[1, 2, 3],                # ← Change these
    time_limit=300,                         # ← Change time limit
    # ... more options
)
```

## Troubleshooting

### "ModuleNotFoundError"

Make sure you're in the right directory and have the virtual environment activated:

```bash
cd aircraft_landing_scheduling
source ../Operations_Guusje/bin/activate  # On Mac/Linux
# OR
..\Operations_Guusje\Scripts\activate     # On Windows
cd code
python run_complete_sensitivity.py
```

### "Out of memory"

If you run out of memory with larger instances, try:
- Reducing the number of aircraft counts
- Reducing the number of runway counts
- Reducing the time limit

### "Takes too long"

If it's taking too long, you can:
- Reduce the time limit (e.g., from 300 to 60 seconds)
- Reduce the aircraft counts (e.g., just [10, 20, 30])
- Press Ctrl+C to cancel (results so far will be saved)

## What to Include in Your Report

From the analysis, you should include:

1. **The comprehensive table** (maybe filtered to show key scenarios)
2. **At least 2-3 heatmaps** (cost, solve time, gap)
3. **Key insights**:
   - How cost decreases with more runways
   - How solve time increases with more aircraft
   - How the gap changes with problem size
   - Comparison with FCFS heuristic (40% improvement!)

## Questions?

If something doesn't work, check:
1. Are you in the right directory?
2. Is the virtual environment activated?
3. Are all dependencies installed? (`pip install -r requirements.txt`)

---

**Good luck with your report, Jelle! 🚀**

_Yo Jelle, je hoeft alleen op deze knop te klikken, super lang te wachten en dan krijg je het resultaat van de sensitivity analysis in een heat map vorm en de desbetreffende super grote tabel van alle individuele runs waarin de solve time, variables, constraints, gap en status allemaal erin staan!_
