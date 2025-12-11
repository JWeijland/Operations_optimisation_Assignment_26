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


# Aircraft type definitions based on 5-category wake turbulence system
@dataclass
class AircraftType:
    """Defines characteristics of different aircraft categories."""
    name: str
    wake_category: str  # B, C, D, E, F
    typical_models: List[str]
    early_cost_per_min: float  # Fuel cost for holding pattern
    late_cost_per_min: float   # Missed connections, gate fees, passenger compensation
    early_buffer_min: float    # Minimum minutes can arrive early
    early_buffer_max: float    # Maximum minutes can arrive early
    late_buffer_min: float     # Minimum minutes can arrive late
    late_buffer_max: float     # Maximum minutes can arrive late


# Define realistic aircraft types - Schiphol uses only 3 categories (B/D/E)
AIRCRAFT_TYPES = {
    'CAT_B': AircraftType(
        name='CAT-B (Upper Heavy)',
        wake_category='B',
        typical_models=['Boeing 747', 'Boeing 777', 'Boeing 787', 'Airbus A330'],
        early_cost_per_min=120.0,  # €120/min fuel cost (largest aircraft)
        late_cost_per_min=250.0,   # €250/min delay cost (most connections)
        early_buffer_min=6.0,      # Upper heavy: most fuel reserve
        early_buffer_max=10.0,     # Can hold longest
        late_buffer_min=12.0,      # Critical connections
        late_buffer_max=18.0       # Highest priority
    ),
    'CAT_C': AircraftType(
        name='CAT-C (Lower Heavy)',
        wake_category='C',
        typical_models=['Not used at Schiphol'],
        early_cost_per_min=100.0,  # Placeholder (not used)
        late_cost_per_min=200.0,
        early_buffer_min=5.0,
        early_buffer_max=8.0,
        late_buffer_min=10.0,
        late_buffer_max=15.0
    ),
    'CAT_D': AircraftType(
        name='CAT-D (Upper Medium)',
        wake_category='D',
        typical_models=['Boeing 737', 'Airbus A320'],
        early_cost_per_min=70.0,   # €70/min fuel cost
        late_cost_per_min=160.0,   # €160/min delay cost
        early_buffer_min=4.0,      # Upper medium: moderate fuel
        early_buffer_max=6.0,
        late_buffer_min=8.0,       # Some flexibility
        late_buffer_max=12.0
    ),
    'CAT_E': AircraftType(
        name='CAT-E (Lower Medium)',
        wake_category='E',
        typical_models=['Embraer 190', 'Embraer 175'],
        early_cost_per_min=50.0,   # €50/min fuel cost
        late_cost_per_min=120.0,   # €120/min delay cost
        early_buffer_min=3.5,      # Lower medium: limited fuel
        early_buffer_max=5.5,
        late_buffer_min=7.0,       # More flexible
        late_buffer_max=10.0
    ),
    'CAT_F': AircraftType(
        name='CAT-F (Light)',
        wake_category='F',
        typical_models=['Not used at Schiphol'],
        early_cost_per_min=40.0,   # Placeholder (not used)
        late_cost_per_min=100.0,
        early_buffer_min=3.0,
        early_buffer_max=5.0,
        late_buffer_min=6.0,
        late_buffer_max=9.0
    )
}


# Wake turbulence separation requirements (in seconds)
# Based on 5-category system with specific separation times
SEPARATION_MATRIX = {
    # CAT-B (Upper Heavy) leading
    ('B', 'B'): 60,   # B behind B
    ('B', 'C'): 60,   # C behind B
    ('B', 'D'): 100,  # D behind B ⚠️
    ('B', 'E'): 120,  # E behind B ⚠️
    ('B', 'F'): 140,  # F behind B ⚠️

    # CAT-C (Lower Heavy) leading
    ('C', 'B'): 60,   # B behind C
    ('C', 'C'): 60,   # C behind C
    ('C', 'D'): 80,   # D behind C ⚠️
    ('C', 'E'): 100,  # E behind C ⚠️
    ('C', 'F'): 120,  # F behind C ⚠️

    # CAT-D (Upper Medium) leading
    ('D', 'B'): 60,   # B behind D
    ('D', 'C'): 60,   # C behind D
    ('D', 'D'): 60,   # D behind D
    ('D', 'E'): 80,   # E behind D ⚠️ (UPDATED!)
    ('D', 'F'): 120,  # F behind D ⚠️

    # CAT-E (Lower Medium) leading
    ('E', 'B'): 60,   # B behind E
    ('E', 'C'): 60,   # C behind E
    ('E', 'D'): 60,   # D behind E
    ('E', 'E'): 60,   # E behind E
    ('E', 'F'): 100,  # F behind E ⚠️

    # CAT-F (Light) leading
    ('F', 'B'): 60,   # B behind F
    ('F', 'C'): 60,   # C behind F
    ('F', 'D'): 60,   # D behind F
    ('F', 'E'): 60,   # E behind F
    ('F', 'F'): 80,   # F behind F ⚠️
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
    num_aircraft: int = 49,  # Changed default to match 5+15+22+6+1=49
    num_runways: int = 1,
    scenario_name: str = "schiphol_evening_1runway",
    cat_b_ratio: float = 0.10,  # CAT-B (Upper Heavy): 10%
    cat_c_ratio: float = 0.31,  # CAT-C (Lower Heavy): 31%
    cat_d_ratio: float = 0.45,  # CAT-D (Upper Medium): 45%
    cat_e_ratio: float = 0.12,  # CAT-E (Lower Medium): 12%
    cat_f_ratio: float = 0.02,  # CAT-F (Light): 2%
    seed: int = 42,
    peak_hour_probability: float = 0.5
) -> ProblemInstance:
    """
    Create realistic Schiphol evening rush hour scenario (18:00-20:00) with 5 aircraft categories.

    Args:
        num_aircraft: Number of arriving aircraft
        num_runways: Number of available runways (1-3)
        scenario_name: Name for the scenario
        cat_b_ratio: Proportion of CAT-B (Upper Heavy) aircraft
        cat_c_ratio: Proportion of CAT-C (Lower Heavy) aircraft
        cat_d_ratio: Proportion of CAT-D (Upper Medium) aircraft
        cat_e_ratio: Proportion of CAT-E (Lower Medium) aircraft
        cat_f_ratio: Proportion of CAT-F (Light) aircraft
        seed: Random seed for reproducibility
        peak_hour_probability: Probability that aircraft arrives in peak hour (0.0-1.0)
                               0.5 = 50% in peak hour (normal)
                               0.9 = 90% in peak hour (extreme congestion!)

    Returns:
        ProblemInstance with realistic Schiphol data
    """
    np.random.seed(seed)

    # Validate ratios
    total_ratio = cat_b_ratio + cat_c_ratio + cat_d_ratio + cat_e_ratio + cat_f_ratio
    assert abs(total_ratio - 1.0) < 0.01, \
        f"Aircraft type ratios must sum to 1.0, got {total_ratio}"

    # Generate aircraft mix - only B, D, E (C and F are 0% at Schiphol)
    num_cat_b = round(num_aircraft * cat_b_ratio)
    num_cat_d = round(num_aircraft * cat_d_ratio)
    num_cat_e = num_aircraft - num_cat_b - num_cat_d  # Remaining aircraft are CAT-E

    # Verify ratios (C and F should be 0)
    if cat_c_ratio > 0.001 or cat_f_ratio > 0.001:
        print(f"WARNING: CAT-C ({cat_c_ratio*100:.1f}%) and CAT-F ({cat_f_ratio*100:.1f}%) should be 0% at Schiphol!")

    # Create aircraft list with only B, D, E
    aircraft_types = (['B'] * num_cat_b +
                     ['D'] * num_cat_d +
                     ['E'] * num_cat_e)
    np.random.shuffle(aircraft_types)

    # Evening rush: arrivals spread over 2 hours (120 minutes)
    # Peak between 19:00-19:30 (minutes 60-90)
    base_time = 0  # Start at minute 0 (18:00)
    time_window = 120  # 2 hours

    aircraft_list = []

    # Create overlapping time windows to force conflicts
    # This simulates realistic runway congestion
    for i, ac_type in enumerate(aircraft_types):
        # Only B, D, E are used at Schiphol
        type_key = {'B': 'CAT_B', 'D': 'CAT_D', 'E': 'CAT_E'}[ac_type]
        type_info = AIRCRAFT_TYPES[type_key]

        # Scheduled Time of Arrival (STA) - peak hour distribution
        # More arrivals during 19:00-19:30 to create congestion
        # User can control this with peak_hour_probability parameter
        if np.random.random() < peak_hour_probability:
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
