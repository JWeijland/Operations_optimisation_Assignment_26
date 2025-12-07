#!/usr/bin/env python3
"""
Fixed Scenario Definitions
4 predefined scenarios with known aircraft and target times
"""

import sys
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "aircraft_landing_scheduling"))

from code.data_loader import Aircraft, ProblemInstance, DataLoader
import numpy as np


def create_scenario_1_single_runway_light():
    """
    Scenario 1: Single Runway - Light Traffic
    - 20 aircraft
    - 1 runway
    - Off-peak traffic pattern
    """
    # Load from existing file
    data_file = Path(__file__).parent.parent / "aircraft_landing_scheduling" / "data" / "schiphol_1runway_light.txt"
    instance = DataLoader.load_from_file(str(data_file))
    return instance


def create_scenario_2_single_runway_heavy():
    """
    Scenario 2: Single Runway - Heavy Traffic
    - 35 aircraft
    - 1 runway
    - Peak hour traffic
    """
    data_file = Path(__file__).parent.parent / "aircraft_landing_scheduling" / "data" / "schiphol_1runway_heavy.txt"
    instance = DataLoader.load_from_file(str(data_file))
    return instance


def create_scenario_3_two_runways():
    """
    Scenario 3: Two Runways - Medium Traffic
    - 30 aircraft
    - 2 runways
    - Medium traffic load
    """
    data_file = Path(__file__).parent.parent / "aircraft_landing_scheduling" / "data" / "schiphol_2runways_medium.txt"
    instance = DataLoader.load_from_file(str(data_file))
    return instance


def create_scenario_4_three_runways():
    """
    Scenario 4: Three Runways - Heavy Traffic
    - 50 aircraft
    - 3 runways
    - Full rush hour
    """
    data_file = Path(__file__).parent.parent / "aircraft_landing_scheduling" / "data" / "schiphol_3runways_heavy.txt"
    instance = DataLoader.load_from_file(str(data_file))
    return instance


# Scenario registry
SCENARIOS = {
    '1': {
        'name': 'Single Runway - Light Traffic',
        'num_aircraft': 20,
        'num_runways': 1,
        'function': create_scenario_1_single_runway_light,
        'description': 'Off-peak traffic, 20 aircraft on single runway'
    },
    '2': {
        'name': 'Single Runway - Heavy Traffic',
        'num_aircraft': 35,
        'num_runways': 1,
        'function': create_scenario_2_single_runway_heavy,
        'description': 'Peak hour traffic, 35 aircraft on single runway'
    },
    '3': {
        'name': 'Two Runways - Medium Traffic',
        'num_aircraft': 30,
        'num_runways': 2,
        'function': create_scenario_3_two_runways,
        'description': 'Medium traffic load across 2 runways'
    },
    '4': {
        'name': 'Three Runways - Heavy Traffic',
        'num_aircraft': 50,
        'num_runways': 3,
        'function': create_scenario_4_three_runways,
        'description': 'Full rush hour with 50 aircraft on 3 runways'
    }
}


def get_scenario(scenario_id):
    """Get scenario by ID (1-4)"""
    if str(scenario_id) not in SCENARIOS:
        raise ValueError(f"Invalid scenario ID: {scenario_id}. Choose 1-4.")

    scenario_info = SCENARIOS[str(scenario_id)]
    instance = scenario_info['function']()

    return instance, scenario_info


def print_scenarios():
    """Print all available scenarios"""
    print("\n" + "=" * 70)
    print("AVAILABLE SCENARIOS")
    print("=" * 70)

    for sid, info in SCENARIOS.items():
        print(f"\n{sid}. {info['name']}")
        print(f"   Aircraft: {info['num_aircraft']}")
        print(f"   Runways: {info['num_runways']}")
        print(f"   {info['description']}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Test: print all scenarios
    print_scenarios()

    # Test: load scenario 1
    print("Testing scenario 1...")
    instance, info = get_scenario(1)
    print(f"✓ Loaded: {info['name']}")
    print(f"  Aircraft count: {instance.num_aircraft}")
