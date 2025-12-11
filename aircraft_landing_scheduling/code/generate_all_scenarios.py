#!/usr/bin/env python3
"""
Generate ALL scenarios with consistent ICAO classification
Volgens WhatsApp Image 2025-12-07 at 16.52.54
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from schiphol_realistic_peak import (
    SEPARATION_MATRIX,
    AIRCRAFT_FLEET,
    COST_STRUCTURE,
    generate_schiphol_peak_hour
)

def generate_all_scenarios():
    """Generate all 4 scenarios with consistent ratios."""

    scenarios = [
        {
            'num_aircraft': 20,
            'num_runways': 1,
            'name': 'schiphol_1runway_light',
            'description': 'Light Traffic - Single Runway',
            'seed': 10
        },
        {
            'num_aircraft': 35,
            'num_runways': 1,
            'name': 'schiphol_1runway_heavy',
            'description': 'Heavy Traffic - Single Runway',
            'seed': 20
        },
        {
            'num_aircraft': 50,
            'num_runways': 3,
            'name': 'schiphol_3runways_heavy',
            'description': 'Heavy Traffic - Three Runways',
            'seed': 30
        },
        {
            'num_aircraft': 50,
            'num_runways': 3,
            'name': 'schiphol_realistic_peak',
            'description': 'Realistic Peak Hour (11:00-11:40)',
            'seed': 42
        }
    ]

    print("="*80)
    print("GENERATING ALL SCENARIOS WITH CONSISTENT ICAO CLASSIFICATION")
    print("="*80)
    print()
    print("Ratio (constant across all scenarios):")
    print("  CAT-B (Upper Heavy):  10%")
    print("  CAT-C (Lower Heavy):  30%")
    print("  CAT-D (Upper Medium): 45%")
    print("  CAT-E (Lower Medium): 12%")
    print("  CAT-F (Light):         3%")
    print()
    print("="*80)
    print()

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. Generating: {scenario['description']}")
        print(f"   Aircraft: {scenario['num_aircraft']}, Runways: {scenario['num_runways']}")

        df, filepath = generate_schiphol_peak_hour(
            num_aircraft=scenario['num_aircraft'],
            num_runways=scenario['num_runways'],
            scenario_name=scenario['name'],
            seed=scenario['seed']
        )

        print(f"   ✓ Generated: {filepath}")

    print()
    print("="*80)
    print("✓ ALL SCENARIOS GENERATED WITH CONSISTENT RATIOS!")
    print("="*80)


if __name__ == "__main__":
    generate_all_scenarios()
