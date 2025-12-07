#!/usr/bin/env python3
"""
Scenario Definitions
Define all test scenarios for the aircraft landing scheduling problem
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "aircraft_landing_scheduling"))

from code.data_loader import DataLoader


# Scenario definitions
SCENARIOS = {
    '1': {
        'name': 'Light Traffic - Single Runway',
        'description': 'Baseline scenario with minimal complexity',
        'num_aircraft': 10,
        'num_runways': 1,
        'data_file': 'data/schiphol_1runway_light.txt'
    },
    '2': {
        'name': 'Heavy Traffic - Single Runway',
        'description': 'Maximum load on single runway',
        'num_aircraft': 20,
        'num_runways': 1,
        'data_file': 'data/schiphol_1runway_heavy.txt'
    },
    '3': {
        'name': 'Heavy Traffic - Three Runways',
        'description': 'Complex scenario with multiple runways',
        'num_aircraft': 50,
        'num_runways': 3,
        'data_file': 'data/schiphol_3runways_heavy.txt'
    },
    '4': {
        'name': 'Extreme Rush Hour',
        'description': 'Same as scenario 3 (alternative configuration)',
        'num_aircraft': 50,
        'num_runways': 3,
        'data_file': 'data/schiphol_3runways_heavy.txt'
    },
    '5': {
        'name': 'Realistic Schiphol Peak Hour',
        'description': 'Realistic scenario based on actual Schiphol data (11:00-11:40 peak)',
        'num_aircraft': 50,
        'num_runways': 3,
        'data_file': 'data/schiphol_realistic_peak.txt'
    }
}


def get_scenario(scenario_id):
    """
    Load a scenario by ID.

    Args:
        scenario_id: Scenario ID (1-5) as string or int

    Returns:
        (ProblemInstance, scenario_info dict)
    """
    scenario_id = str(scenario_id)

    if scenario_id not in SCENARIOS:
        raise ValueError(f"Unknown scenario ID: {scenario_id}. Valid options: 1-5")

    info = SCENARIOS[scenario_id]

    # Load problem instance
    data_file = Path(__file__).parent.parent / info['data_file']

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    instance = DataLoader.load_from_file(str(data_file))

    return instance, info


def print_scenarios():
    """Print available scenarios."""
    print("\n" + "=" * 70)
    print("AVAILABLE SCENARIOS")
    print("=" * 70)

    for scenario_id, info in SCENARIOS.items():
        print(f"\nScenario {scenario_id}: {info['name']}")
        print(f"  Aircraft: {info['num_aircraft']}")
        print(f"  Runways: {info['num_runways']}")
        print(f"  Description: {info['description']}")
        print(f"  Data: {info['data_file']}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    # Test scenario loading
    print("Testing scenario definitions...")

    for scenario_id in ['1', '2', '3', '4', '5']:
        try:
            instance, info = get_scenario(scenario_id)
            print(f"✓ Scenario {scenario_id}: {info['name']} - {len(instance.aircraft)} aircraft loaded")
        except Exception as e:
            print(f"✗ Scenario {scenario_id}: {e}")
