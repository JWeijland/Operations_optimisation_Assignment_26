#!/usr/bin/env python3
"""
Schiphol Realistic Peak Hour Scenario Generator
Gebaseerd op echte Schiphol data en ICAO wake turbulence categories

Data bronnen:
- Schiphol vliegtuig mix (WhatsApp foto's)
- ICAO wake turbulence separation matrix
- Piektijd: 11:00-11:40 (40 minuten)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

# ICAO Wake Turbulence Categories
# CAT-A = Super Heavy (A380) - SKIP
# CAT-B = Upper Heavy (B747, B777, A330, A340, A350)
# CAT-C = Lower Heavy (B757, B767, A300)
# CAT-D = Upper Medium (B737, A320, A319)
# CAT-E = Lower Medium (B717, ATR-42/72)
# CAT-F = Light (Falcon, Citation, Beechjet)

# Separation matrix in MINUTES (converted from ICAO seconds)
# Original ICAO values in seconds, divided by 60 for minutes
SEPARATION_MATRIX = {
    # Leader CAT-B (Upper Heavy) → Follower (100-180s → 1.67-3.00min)
    'CAT-B': {'CAT-B': 1.67, 'CAT-C': 2.00, 'CAT-D': 2.33, 'CAT-E': 2.67, 'CAT-F': 3.00},

    # Leader CAT-C (Lower Heavy) → Follower (80-120s → 1.33-2.00min)
    'CAT-C': {'CAT-B': 1.33, 'CAT-C': 1.33, 'CAT-D': 1.67, 'CAT-E': 2.00, 'CAT-F': 2.00},

    # Leader CAT-D (Upper Medium) → Follower (60-120s → 1.00-2.00min)
    'CAT-D': {'CAT-B': 1.00, 'CAT-C': 1.00, 'CAT-D': 1.33, 'CAT-E': 1.67, 'CAT-F': 2.00},

    # Leader CAT-E (Lower Medium) → Follower (60-100s → 1.00-1.67min)
    'CAT-E': {'CAT-B': 1.00, 'CAT-C': 1.00, 'CAT-D': 1.00, 'CAT-E': 1.67, 'CAT-F': 1.67},

    # Leader CAT-F (Light) → Follower (60-80s → 1.00-1.33min)
    'CAT-F': {'CAT-B': 1.00, 'CAT-C': 1.00, 'CAT-D': 1.00, 'CAT-E': 1.00, 'CAT-F': 1.33},
}

# Realistische Schiphol vliegtuig types en hun dagelijkse frequentie
AIRCRAFT_FLEET = {
    # CAT-B: Upper Heavy (20% van traffic)
    'CAT-B': [
        ('B747-400', 20),      # 20 per dag
        ('B777-300ER', 40),    # 40 per dag
        ('A330-300', 29),      # 29 per dag
        ('A340-300', 10),      # Minder frequent
    ],

    # CAT-C: Lower Heavy (15% van traffic)
    'CAT-C': [
        ('B767-300', 25),      # Cargo
        ('B757-200', 15),      # KLM/Transavia
        ('A300-600', 10),      # Cargo
    ],

    # CAT-D: Upper Medium (50% van traffic - GROOTSTE GROEP)
    'CAT-D': [
        ('B737-800', 226),     # 226 per dag - MEESTE!
        ('A320-200', 95),      # 95 per dag
        ('A321-200', 40),      # 40 per dag
        ('A319-100', 30),      # 30 per dag
    ],

    # CAT-E: Lower Medium (10% van traffic)
    'CAT-E': [
        ('ATR-72', 50),        # Regionaal
        ('B717-200', 20),      # Regionaal
        ('E190', 30),          # Embraer
    ],

    # CAT-F: Light (5% van traffic)
    'CAT-F': [
        ('Falcon 20', 20),     # Business jets
        ('Citation', 30),      # Business jets
        ('Beechjet', 15),      # Business jets
    ],
}

# Cost structure per category (EUR per minuut)
COST_STRUCTURE = {
    'CAT-B': {'early': 120, 'late': 250},  # Heavy = duur
    'CAT-C': {'early': 100, 'late': 200},
    'CAT-D': {'early': 60, 'late': 150},   # Medium = normaal
    'CAT-E': {'early': 40, 'late': 100},
    'CAT-F': {'early': 30, 'late': 80},    # Light = goedkoop
}


def generate_schiphol_peak_hour(
    num_aircraft: int = 50,
    num_runways: int = 3,
    scenario_name: str = "schiphol_realistic_peak",
    seed: int = 42
) -> Tuple[pd.DataFrame, str]:
    """
    Genereer realistisch Schiphol peak hour scenario.

    Piektijd: 11:00-11:40 (40 minuten)
    Time window: 10:00-12:00 (2 uur totaal)

    Args:
        num_aircraft: Aantal vliegtuigen (default 50)
        num_runways: Aantal runways (1-3, Schiphol heeft 6 maar we simuleren 3)
        scenario_name: Naam voor output file
        seed: Random seed voor reproduceerbaarheid

    Returns:
        (DataFrame met vliegtuig data, output filepath)
    """
    np.random.seed(seed)

    # Verdeling volgens realistische Schiphol mix
    category_distribution = {
        'CAT-B': 0.20,  # 20% heavy
        'CAT-C': 0.15,  # 15% heavy
        'CAT-D': 0.50,  # 50% medium (MEESTE)
        'CAT-E': 0.10,  # 10% medium
        'CAT-F': 0.05,  # 5% light
    }

    # Genereer vliegtuigen
    aircraft_list = []
    aircraft_id = 1

    for category, ratio in category_distribution.items():
        n_category = int(num_aircraft * ratio)

        for _ in range(n_category):
            # Kies random vliegtuigtype uit deze categorie
            aircraft_types = AIRCRAFT_FLEET[category]
            weights = [freq for _, freq in aircraft_types]
            aircraft_type = np.random.choice(
                [name for name, _ in aircraft_types],
                p=np.array(weights) / sum(weights)
            )

            # Genereer target tijd
            # 70% in peak hour (11:00-11:40 = minuut 60-100)
            # 30% buiten peak (10:00-11:00 en 11:40-12:00)
            if np.random.random() < 0.70:
                # Peak hour: 11:00-11:40
                target_time = np.random.uniform(60, 100)
            else:
                # Off-peak
                if np.random.random() < 0.5:
                    target_time = np.random.uniform(0, 60)  # 10:00-11:00
                else:
                    target_time = np.random.uniform(100, 120)  # 11:40-12:00

            # Time window: +/- 25 minuten voor heavy, +/- 20 voor medium/light
            # (Wider windows to ensure feasibility with realistic separation times)
            if category in ['CAT-B', 'CAT-C']:
                window = 25
            else:
                window = 20

            earliest = max(0, target_time - window)
            latest = min(120, target_time + window)

            # Costs
            early_cost = COST_STRUCTURE[category]['early']
            late_cost = COST_STRUCTURE[category]['late']

            aircraft_list.append({
                'id': aircraft_id,
                'type': aircraft_type,
                'category': category,
                'earliest': round(earliest, 2),
                'target': round(target_time, 2),
                'latest': round(latest, 2),
                'early_penalty': early_cost,
                'late_penalty': late_cost,
            })

            aircraft_id += 1

    # Shuffle aircraft (mix volgorde)
    np.random.shuffle(aircraft_list)

    # Re-assign sequential IDs
    for i, ac in enumerate(aircraft_list, 1):
        ac['id'] = i

    # Create DataFrame
    df = pd.DataFrame(aircraft_list)

    # Bereken separation matrix voor dit scenario
    separation_data = []
    for i in range(len(df)):
        leader_cat = df.iloc[i]['category']
        for j in range(len(df)):
            follower_cat = df.iloc[j]['category']
            if i != j:
                sep_time = SEPARATION_MATRIX[leader_cat][follower_cat]
                separation_data.append({
                    'leader_id': i + 1,
                    'follower_id': j + 1,
                    'leader_cat': leader_cat,
                    'follower_cat': follower_cat,
                    'separation_sec': sep_time
                })

    sep_df = pd.DataFrame(separation_data)

    # Write to file (Beasley format)
    output_file = f"data/{scenario_name}.txt"
    write_beasley_format(df, sep_df, num_runways, output_file)

    # Also write to Excel for easy viewing
    excel_file = f"data/{scenario_name}.xlsx"
    write_excel_overview(df, sep_df, excel_file)

    print(f"\n✓ Realistic Schiphol scenario generated:")
    print(f"  Aircraft: {len(df)}")
    print(f"  Runways: {num_runways}")
    print(f"  Category distribution:")
    for cat in ['CAT-B', 'CAT-C', 'CAT-D', 'CAT-E', 'CAT-F']:
        count = len(df[df['category'] == cat])
        print(f"    {cat}: {count} ({count/len(df)*100:.0f}%)")
    print(f"  Peak hour (60-100 min): {len(df[(df['target'] >= 60) & (df['target'] <= 100)])} aircraft")
    print(f"  Output: {output_file}")
    print(f"  Excel: {excel_file}")

    return df, output_file


def write_beasley_format(df: pd.DataFrame, sep_df: pd.DataFrame, num_runways: int, filepath: str):
    """Write in OR-Library format (Beasley et al. 2000) for compatibility."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        # Header: num_aircraft freeze_time
        f.write(f"{len(df)} 0.0\n")

        # Aircraft data with inline separation matrix
        # Format: E T L g h S_i1 S_i2 ... S_in
        for i, row in df.iterrows():
            # Aircraft parameters
            f.write(f"{row['earliest']:.2f} {row['target']:.2f} {row['latest']:.2f} ")
            f.write(f"{row['early_penalty']} {row['late_penalty']}")

            # Separation times when this aircraft (i) lands before others (j)
            for j in range(len(df)):
                leader_cat = row['category']
                follower_cat = df.iloc[j]['category']
                sep_time = SEPARATION_MATRIX[leader_cat][follower_cat]
                f.write(f" {sep_time}")

            # Add comment with aircraft info
            f.write(f"  # {row['id']} {row['type']} {row['category']}\n")


def write_excel_overview(df: pd.DataFrame, sep_df: pd.DataFrame, filepath: str):
    """Write overzichtelijke Excel met tabs."""
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Tab 1: Aircraft Overview
        overview = df[['id', 'type', 'category', 'earliest', 'target', 'latest',
                      'early_penalty', 'late_penalty']].copy()
        overview.columns = ['ID', 'Aircraft Type', 'Wake Category', 'Earliest (min)',
                           'Target (min)', 'Latest (min)', 'Early Cost (€/min)', 'Late Cost (€/min)']
        overview.to_excel(writer, sheet_name='Aircraft', index=False)

        # Tab 2: Separation Matrix (pivot format)
        sep_pivot = sep_df.pivot_table(
            index='leader_cat',
            columns='follower_cat',
            values='separation_sec',
            aggfunc='first'
        )
        sep_pivot.to_excel(writer, sheet_name='Separation Matrix')

        # Tab 3: Category Summary
        summary = df.groupby('category').agg({
            'id': 'count',
            'target': ['min', 'mean', 'max'],
            'early_penalty': 'first',
            'late_penalty': 'first'
        }).round(2)
        summary.to_excel(writer, sheet_name='Category Summary')

        # Tab 4: Detailed Separations (first 100 rows voor readability)
        sep_detail = sep_df.head(100).copy()
        sep_detail.columns = ['Leader ID', 'Follower ID', 'Leader Cat',
                             'Follower Cat', 'Separation (sec)']
        sep_detail.to_excel(writer, sheet_name='Separation Details', index=False)


if __name__ == "__main__":
    # Generate scenario
    df, filepath = generate_schiphol_peak_hour(
        num_aircraft=50,
        num_runways=3,
        scenario_name="schiphol_realistic_peak",
        seed=42
    )

    print("\n✓ Realistic Schiphol peak hour scenario created!")
    print("\nExample aircraft:")
    print(df.head(10).to_string())
