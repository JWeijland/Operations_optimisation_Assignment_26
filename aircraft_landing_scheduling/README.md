# Aircraft Landing Scheduling - Static Case

Implementation of the aircraft landing scheduling problem based on Beasley et al. (2000).

## Overview

This project implements a mixed-integer programming approach to optimally schedule aircraft landings at an airport, considering:
- Time windows for each aircraft
- Separation requirements between landings
- Single and multiple runway scenarios
- Cost minimization objective

## Project Structure

```
├── code/                  # Source code
├── data/                  # Test instances (OR-Library format)
├── results/              # Output tables and figures
└── report/               # Technical report
```

## Requirements

- Python 3.8+
- PuLP (or Gurobi if available)
- NumPy, Pandas, Matplotlib
- See requirements.txt for full list

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run all experiments:
```bash
python code/main.py --batch --data data/
```

### Run specific instance:
```bash
python code/main.py --instance data/airland1.txt --runways 1
```

### Compare heuristic vs optimal:
```bash
python code/main.py --instance sample --runways 1
```

## Data Format

Input files follow OR-Library format with aircraft data and separation matrices.

## Results

Results are saved in:
- `results/tables/` - CSV files with computational results
- `results/figures/` - Gantt charts and performance plots

## Authors

[Your Name]
[Student ID]
Delft University of Technology

## References

Beasley, J.E., Krishnamoorthy, M., Sharaiha, Y.M. and Abramson, D. (2000).
"Scheduling aircraft landings—the static case." Transportation Science, 34(2), 180-197.
