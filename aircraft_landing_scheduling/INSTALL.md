# Installatie Instructies

## Stap 1: Maak een virtual environment aan

```bash
cd /Users/jelleweijland/Documents/AE\ Vakken/Operations_Assignment/Operations_optimisation_Assignment_26/
python3 -m venv Operations_Guusje
```

## Stap 2: Activeer de virtual environment

```bash
source Operations_Guusje/bin/activate
```

## Stap 3: Installeer alle vereiste packages

```bash
cd aircraft_landing_scheduling
pip install -r requirements.txt
```

Of installeer packages één voor één:
```bash
pip install numpy pandas matplotlib pulp tabulate scipy seaborn openpyxl
```

## Stap 4: Verifieer de installatie

```bash
python3 -c "import openpyxl; print('✓ All packages installed')"
```

## Stap 5: Run het programma

```bash
python code/main.py
```

## Vereiste Packages

- **numpy**: Numerieke berekeningen
- **pandas**: Data manipulatie en analysis
- **matplotlib**: Grafieken en visualisaties
- **pulp**: Linear programming solver (voor optimale oplossing)
- **tabulate**: Mooie tabellen in terminal
- **scipy**: Wetenschappelijke berekeningen
- **seaborn**: Mooie visualisaties
- **openpyxl**: Excel export functionaliteit

## Troubleshooting

### Package niet gevonden
```bash
pip install <package-naam>
```

### Virtual environment deactiveren
```bash
deactivate
```

### Virtual environment opnieuw activeren
```bash
source /Users/jelleweijland/Documents/AE\ Vakken/Operations_Assignment/Operations_optimisation_Assignment_26/Operations_Guusje/bin/activate
```
