#!/usr/bin/env python3
"""
Realistic Schiphol Airport Landing Scenarios
Based on real-world aircraft types, separation requirements, and peak hour traffic
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    from .data_loader import Aircraft, ProblemInstance, DataLoader
except ImportError:
    from data_loader import Aircraft, ProblemInstance, DataLoader


# Aircraft type definitions based on ICAO wake turbulence categories
@dataclass
class AircraftType:
    """Defines characteristics of different aircraft categories."""
    name: str
    wake_category: str  # Heavy, Medium, Light
    typical_models: List[str]
    early_cost_per_min: float  # Fuel cost for holding pattern
    late_cost_per_min: float   # Missed connections, gate fees, passenger compensation
    early_buffer_min: float    # Minimum minutes can arrive early
    early_buffer_max: float    # Maximum minutes can arrive early
    late_buffer_min: float     # Minimum minutes can arrive late
    late_buffer_max: float     # Maximum minutes can arrive late


# Define realistic aircraft types with consistent time buffers per type
AIRCRAFT_TYPES = {
    'HEAVY': AircraftType(
        name='Heavy',
        wake_category='H',
        typical_models=['Boeing 747', 'Boeing 777', 'Airbus A330', 'Airbus A350'],
        early_cost_per_min=100.0,  # €100/min fuel cost
        late_cost_per_min=200.0,   # €200/min delay cost (higher due to connections)
        early_buffer_min=5.0,      # Heavy aircraft: more fuel reserve
        early_buffer_max=8.0,      # Can hold longer
        late_buffer_min=10.0,      # More critical connections
        late_buffer_max=15.0       # Higher priority for on-time
    ),
    'MEDIUM': AircraftType(
        name='Medium',
        wake_category='M',
        typical_models=['Boeing 737', 'Airbus A320', 'Embraer E190'],
        early_cost_per_min=60.0,   # €60/min fuel cost
        late_cost_per_min=150.0,   # €150/min delay cost
        early_buffer_min=4.0,      # Medium aircraft: moderate fuel
        early_buffer_max=6.0,
        late_buffer_min=8.0,       # Some flexibility
        late_buffer_max=12.0
    ),
    'LIGHT': AircraftType(
        name='Light',
        wake_category='L',
        typical_models=['Cessna Citation', 'Embraer Phenom', 'Pilatus PC-12'],
        early_cost_per_min=30.0,   # €30/min fuel cost
        late_cost_per_min=80.0,    # €80/min delay cost
        early_buffer_min=3.0,      # Light aircraft: less fuel reserve
        early_buffer_max=5.0,      # Tighter constraints
        late_buffer_min=6.0,       # Lower priority, more flexible
        late_buffer_max=10.0
    )
}


# Wake turbulence separation requirements (in seconds)
# Based on ICAO standards
SEPARATION_MATRIX = {
    ('H', 'H'): 90,   # Heavy behind Heavy
    ('H', 'M'): 120,  # Medium behind Heavy
    ('H', 'L'): 180,  # Light behind Heavy
    ('M', 'H'): 60,   # Heavy behind Medium
    ('M', 'M'): 60,   # Medium behind Medium
    ('M', 'L'): 120,  # Light behind Medium
    ('L', 'H'): 60,   # Heavy behind Light
    ('L', 'M'): 60,   # Medium behind Light
    ('L', 'L'): 60,   # Light behind Light
}


def get_separation_time(leading_type: str, following_type: str) -> float:
    """
    Get required separation time when 'following' aircraft lands after 'leading' aircraft.

    Args:
        leading_type: Wake category of leading aircraft (H/M/L)
        following_type: Wake category of following aircraft (H/M/L)

    Returns:
        Separation time in seconds
    """
    return SEPARATION_MATRIX.get((leading_type, following_type), 60)


def create_schiphol_evening_rush(
    num_aircraft: int = 30,
    num_runways: int = 1,
    scenario_name: str = "schiphol_evening_1runway",
    heavy_ratio: float = 0.3,
    medium_ratio: float = 0.6,
    light_ratio: float = 0.1,
    seed: int = 42
) -> ProblemInstance:
    """
    Create realistic Schiphol evening rush hour scenario (18:00-20:00).

    Args:
        num_aircraft: Number of arriving aircraft
        num_runways: Number of available runways (1-3)
        scenario_name: Name for the scenario
        heavy_ratio: Proportion of heavy aircraft
        medium_ratio: Proportion of medium aircraft
        light_ratio: Proportion of light aircraft
        seed: Random seed for reproducibility

    Returns:
        ProblemInstance with realistic Schiphol data
    """
    np.random.seed(seed)

    # Validate ratios
    assert abs(heavy_ratio + medium_ratio + light_ratio - 1.0) < 0.01, \
        "Aircraft type ratios must sum to 1.0"

    # Generate aircraft mix
    num_heavy = int(num_aircraft * heavy_ratio)
    num_medium = int(num_aircraft * medium_ratio)
    num_light = num_aircraft - num_heavy - num_medium

    aircraft_types = (['H'] * num_heavy +
                     ['M'] * num_medium +
                     ['L'] * num_light)
    np.random.shuffle(aircraft_types)

    # Evening rush: arrivals spread over 2 hours (120 minutes)
    # Peak between 19:00-19:30 (minutes 60-90)
    base_time = 0  # Start at minute 0 (18:00)
    time_window = 120  # 2 hours

    aircraft_list = []

    # Create overlapping time windows to force conflicts
    # This simulates realistic runway congestion
    for i, ac_type in enumerate(aircraft_types):
        type_key = {'H': 'HEAVY', 'M': 'MEDIUM', 'L': 'LIGHT'}[ac_type]
        type_info = AIRCRAFT_TYPES[type_key]

        # Scheduled Time of Arrival (STA) - peak hour distribution
        # More arrivals during 19:00-19:30 to create congestion
        if np.random.random() < 0.5:  # 50% chance of peak hour (increased!)
            sta = np.random.uniform(60, 90)  # Peak: minute 60-90 (19:00-19:30)
        else:
            sta = np.random.uniform(0, 120)  # Rest spread over 2 hours

        # REALISTIC VARIABILITY PER AIRCRAFT TYPE:
        # Time buffers are consistent per aircraft type
        # Heavy: more fuel, more flexibility
        # Medium: moderate constraints
        # Light: tight fuel constraints
        early_buffer = np.random.uniform(
            type_info.early_buffer_min,
            type_info.early_buffer_max
        )
        late_buffer = np.random.uniform(
            type_info.late_buffer_min,
            type_info.late_buffer_max
        )

        earliest_time = max(0, sta - early_buffer)
        latest_time = sta + late_buffer

        # Add stochastic variation: actual preferred time might differ slightly from STA
        # This represents uncertainty in flight arrival predictions
        # Shifts target time by -2 to +2 minutes randomly
        target_variation = np.random.uniform(-2, 2)
        actual_target = np.clip(sta + target_variation, earliest_time + 0.5, latest_time - 0.5)

        # Convert costs from per-minute to per-time-unit
        # Since we're using minutes as time units now
        early_penalty = type_info.early_cost_per_min
        late_penalty = type_info.late_cost_per_min

        aircraft = Aircraft(
            id=i + 1,
            appearance_time=earliest_time,
            target_time=actual_target,  # Use varied target instead of exact STA
            latest_time=latest_time,
            early_penalty=early_penalty,
            late_penalty=late_penalty
        )

        # Store aircraft type as metadata (we'll use it for separation matrix)
        aircraft._wake_category = ac_type

        aircraft_list.append(aircraft)

    # Build separation matrix based on wake turbulence
    n = len(aircraft_list)
    separation_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                leading_type = aircraft_list[i]._wake_category
                following_type = aircraft_list[j]._wake_category
                # Convert seconds to minutes (our time unit)
                sep_seconds = get_separation_time(leading_type, following_type)
                sep_minutes = sep_seconds / 60.0
                separation_matrix[i, j] = sep_minutes

    instance = ProblemInstance(
        aircraft=aircraft_list,
        separation_matrix=separation_matrix,
        freeze_time=0.0
    )

    # Save to file
    output_path = Path("data") / f"{scenario_name}.txt"
    _save_schiphol_scenario(instance, output_path, scenario_name, aircraft_types, num_runways)

    return instance


def _save_schiphol_scenario(
    instance: ProblemInstance,
    filepath: Path,
    scenario_name: str,
    aircraft_types: List[str],
    num_runways: int
):
    """Save Schiphol scenario to file with detailed metadata."""

    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Count aircraft by type
    type_counts = {
        'H': aircraft_types.count('H'),
        'M': aircraft_types.count('M'),
        'L': aircraft_types.count('L')
    }

    with open(filepath, 'w') as f:
        # Write header with metadata
        f.write(f"# Schiphol Airport Landing Scenario - {scenario_name}\n")
        f.write(f"# Realistic evening rush hour (18:00-20:00)\n")
        f.write(f"# {instance.num_aircraft} aircraft, {num_runways} runway(s)\n")
        f.write(f"# Aircraft mix: {type_counts['H']} Heavy, {type_counts['M']} Medium, {type_counts['L']} Light\n")
        f.write(f"# Time unit: MINUTES from 18:00 (base time)\n")
        f.write(f"# Costs: EUR per MINUTE early/late\n")
        f.write(f"# Separation: Based on ICAO wake turbulence categories\n")
        f.write("#\n")
        f.write("# Format: num_aircraft freeze_time\n")
        f.write("# For each aircraft: appearance_time target_time latest_time early_penalty late_penalty sep_1 sep_2 ... sep_n\n")
        f.write(f"# Aircraft types (in order): {' '.join(aircraft_types)}\n")

        # Write data
        f.write(f"{instance.num_aircraft} {instance.freeze_time}\n")

        for i, aircraft in enumerate(instance.aircraft):
            ac_type = aircraft._wake_category
            type_key = {'H': 'HEAVY', 'M': 'MEDIUM', 'L': 'LIGHT'}[ac_type]
            type_name = AIRCRAFT_TYPES[type_key].name

            # Basic data
            f.write(f"{aircraft.appearance_time:.1f} ")
            f.write(f"{aircraft.target_time:.1f} ")
            f.write(f"{aircraft.latest_time:.1f} ")
            f.write(f"{aircraft.early_penalty:.0f} ")
            f.write(f"{aircraft.late_penalty:.0f}")

            # Separation times
            for j in range(instance.num_aircraft):
                sep = instance.get_separation(i, j)
                f.write(f" {sep:.1f}")

            f.write(f"  # A{aircraft.id} ({type_name})\n")

    print(f"Schiphol scenario saved to: {filepath}")


# Predefined realistic scenarios
def create_scenario_1runway_light() -> ProblemInstance:
    """Scenario 1: Single runway, light traffic (20 aircraft, off-peak)."""
    return create_schiphol_evening_rush(
        num_aircraft=20,
        num_runways=1,
        scenario_name="schiphol_1runway_light",
        heavy_ratio=0.25,
        medium_ratio=0.65,
        light_ratio=0.10,
        seed=100
    )


def create_scenario_1runway_heavy() -> ProblemInstance:
    """Scenario 2: Single runway, heavy traffic (35 aircraft, peak hour)."""
    return create_schiphol_evening_rush(
        num_aircraft=35,
        num_runways=1,
        scenario_name="schiphol_1runway_heavy",
        heavy_ratio=0.35,
        medium_ratio=0.55,
        light_ratio=0.10,
        seed=101
    )


def create_scenario_2runways_medium() -> ProblemInstance:
    """Scenario 3: Two runways, medium traffic (30 aircraft)."""
    return create_schiphol_evening_rush(
        num_aircraft=30,
        num_runways=2,
        scenario_name="schiphol_2runways_medium",
        heavy_ratio=0.30,
        medium_ratio=0.60,
        light_ratio=0.10,
        seed=102
    )


def create_scenario_3runways_heavy() -> ProblemInstance:
    """Scenario 4: Three runways, heavy traffic (50 aircraft, full rush hour)."""
    return create_schiphol_evening_rush(
        num_aircraft=50,
        num_runways=3,
        scenario_name="schiphol_3runways_heavy",
        heavy_ratio=0.35,
        medium_ratio=0.55,
        light_ratio=0.10,
        seed=103
    )


if __name__ == "__main__":
    """Generate all predefined scenarios."""
    print("Generating realistic Schiphol landing scenarios...")
    print("=" * 70)

    scenarios = [
        ("Scenario 1: Single runway, light traffic", create_scenario_1runway_light),
        ("Scenario 2: Single runway, heavy traffic", create_scenario_1runway_heavy),
        ("Scenario 3: Two runways, medium traffic", create_scenario_2runways_medium),
        ("Scenario 4: Three runways, heavy traffic", create_scenario_3runways_heavy),
    ]

    for desc, create_func in scenarios:
        print(f"\n{desc}")
        instance = create_func()
        print(f"  ✓ Generated {instance.num_aircraft} aircraft")

    print("\n" + "=" * 70)
    print("All scenarios generated successfully!")
    print("\nYou can now run them with:")
    print("  python code/main.py --instance data/schiphol_1runway_light.txt --runways 1")
    print("  python code/main.py --instance data/schiphol_2runways_medium.txt --runways 2")
    print("  etc.")
