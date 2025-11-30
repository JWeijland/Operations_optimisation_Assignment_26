#!/usr/bin/env python3
"""
Utility Functions for Aircraft Landing Scheduling Project
Helper functions for formatting, exporting, and analysis
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from tabulate import tabulate

from .data_loader import ProblemInstance, Aircraft
from .model import Solution


def format_time(seconds: float) -> str:
    """
    Format time in human-readable format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_cost(cost: float) -> str:
    """
    Format cost with appropriate precision.

    Args:
        cost: Cost value

    Returns:
        Formatted cost string
    """
    if cost < 10:
        return f"{cost:.4f}"
    elif cost < 100:
        return f"{cost:.2f}"
    else:
        return f"{cost:.1f}"


def create_solution_summary(
    instance: ProblemInstance,
    solution: Solution
) -> Dict[str, Any]:
    """
    Create comprehensive summary of a solution.

    Args:
        instance: Problem instance
        solution: Solution to summarize

    Returns:
        Dictionary with summary statistics
    """
    # Calculate deviations
    early_count = 0
    late_count = 0
    on_time_count = 0
    total_early_time = 0.0
    total_late_time = 0.0
    max_early = 0.0
    max_late = 0.0

    for aircraft in instance.aircraft:
        landing_time = solution.get_landing_time(aircraft.id)
        deviation = landing_time - aircraft.target_time

        if abs(deviation) < 0.01:
            on_time_count += 1
        elif deviation < 0:
            early_count += 1
            total_early_time += abs(deviation)
            max_early = max(max_early, abs(deviation))
        else:
            late_count += 1
            total_late_time += deviation
            max_late = max(max_late, deviation)

    # Calculate runway utilization
    runway_counts = {}
    for runway in solution.runway_assignments.values():
        runway_counts[runway] = runway_counts.get(runway, 0) + 1

    summary = {
        'num_aircraft': instance.num_aircraft,
        'num_runways': len(set(solution.runway_assignments.values())),
        'total_cost': solution.objective_value,
        'solve_time': solution.solve_time,
        'status': solution.status,
        'early_count': early_count,
        'on_time_count': on_time_count,
        'late_count': late_count,
        'avg_early_time': total_early_time / early_count if early_count > 0 else 0,
        'avg_late_time': total_late_time / late_count if late_count > 0 else 0,
        'max_early_time': max_early,
        'max_late_time': max_late,
        'runway_utilization': runway_counts
    }

    return summary


def print_solution_details(
    instance: ProblemInstance,
    solution: Solution,
    verbose: bool = True
):
    """
    Print detailed solution information.

    Args:
        instance: Problem instance
        solution: Solution to print
        verbose: Include detailed aircraft information
    """
    print("\n" + "="*70)
    print("SOLUTION DETAILS")
    print("="*70)

    # Summary stats
    summary = create_solution_summary(instance, solution)

    print(f"\nGeneral Information:")
    print(f"  Aircraft:        {summary['num_aircraft']}")
    print(f"  Runways:         {summary['num_runways']}")
    print(f"  Total Cost:      {format_cost(summary['total_cost'])}")
    print(f"  Solve Time:      {format_time(summary['solve_time'])}")
    print(f"  Status:          {summary['status']}")

    print(f"\nLanding Statistics:")
    print(f"  On-time:         {summary['on_time_count']} aircraft")
    print(f"  Early:           {summary['early_count']} aircraft "
          f"(avg: {summary['avg_early_time']:.2f}, max: {summary['max_early_time']:.2f})")
    print(f"  Late:            {summary['late_count']} aircraft "
          f"(avg: {summary['avg_late_time']:.2f}, max: {summary['max_late_time']:.2f})")

    print(f"\nRunway Utilization:")
    for runway, count in sorted(summary['runway_utilization'].items()):
        percentage = (count / summary['num_aircraft']) * 100
        print(f"  Runway {runway}:      {count} aircraft ({percentage:.1f}%)")

    if verbose:
        print(f"\nDetailed Schedule:")
        print("-"*70)

        # Create table data
        table_data = []
        for aircraft in sorted(instance.aircraft, key=lambda a: a.id):
            landing_time = solution.get_landing_time(aircraft.id)
            runway = solution.get_runway(aircraft.id)
            deviation = landing_time - aircraft.target_time
            cost = aircraft.calculate_cost(landing_time)

            status = "On-time"
            if deviation < -0.01:
                status = f"Early {abs(deviation):.2f}"
            elif deviation > 0.01:
                status = f"Late {deviation:.2f}"

            table_data.append([
                f"A{aircraft.id}",
                runway,
                f"{aircraft.appearance_time:.1f}",
                f"{aircraft.target_time:.1f}",
                f"{aircraft.latest_time:.1f}",
                f"{landing_time:.1f}",
                status,
                f"{cost:.2f}"
            ])

        headers = ["Aircraft", "Runway", "E", "T", "L", "Landing", "Status", "Cost"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    print("="*70 + "\n")


def export_solution_json(
    instance: ProblemInstance,
    solution: Solution,
    filepath: str
):
    """
    Export solution to JSON format.

    Args:
        instance: Problem instance
        solution: Solution to export
        filepath: Output file path
    """
    # Create export data
    data = {
        'metadata': {
            'num_aircraft': instance.num_aircraft,
            'num_runways': len(set(solution.runway_assignments.values())),
            'total_cost': solution.objective_value,
            'solve_time': solution.solve_time,
            'status': solution.status
        },
        'aircraft': []
    }

    for aircraft in instance.aircraft:
        landing_time = solution.get_landing_time(aircraft.id)
        runway = solution.get_runway(aircraft.id)

        aircraft_data = {
            'id': aircraft.id,
            'appearance_time': aircraft.appearance_time,
            'target_time': aircraft.target_time,
            'latest_time': aircraft.latest_time,
            'landing_time': landing_time,
            'runway': runway,
            'deviation': landing_time - aircraft.target_time,
            'cost': aircraft.calculate_cost(landing_time)
        }
        data['aircraft'].append(aircraft_data)

    # Write to file
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Solution exported to: {filepath}")


def create_latex_table(
    df: pd.DataFrame,
    caption: str = "",
    label: str = ""
) -> str:
    """
    Convert DataFrame to LaTeX table format.

    Args:
        df: DataFrame to convert
        caption: Table caption
        label: LaTeX label for referencing

    Returns:
        LaTeX table string
    """
    # Format numeric columns
    df_formatted = df.copy()
    for col in df_formatted.columns:
        if df_formatted[col].dtype in [np.float64, np.float32]:
            if 'Time' in col:
                df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.4f}")
            elif 'Cost' in col:
                df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.2f}")
            elif 'Gap' in col or '%' in col:
                df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.2f}")
            else:
                df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.2f}")

    # Generate LaTeX
    latex = df_formatted.to_latex(
        index=False,
        escape=False,
        column_format='l' + 'r' * (len(df.columns) - 1)
    )

    # Add caption and label
    if caption:
        latex = latex.replace(
            r'\end{tabular}',
            f'\\end{{tabular}}\n\\caption{{{caption}}}'
        )
    if label:
        latex = latex.replace(
            r'\end{table}',
            f'\\label{{{label}}}\n\\end{{table}}'
        )

    return latex


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate summary statistics for a list of values.

    Args:
        values: List of numeric values

    Returns:
        Dictionary with statistics
    """
    if not values:
        return {
            'mean': 0, 'median': 0, 'std': 0,
            'min': 0, 'max': 0, 'range': 0
        }

    return {
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'range': np.max(values) - np.min(values)
    }


def print_table(
    data: List[List[Any]],
    headers: List[str],
    title: Optional[str] = None,
    tablefmt: str = "grid"
):
    """
    Print formatted ASCII table.

    Args:
        data: Table data as list of rows
        headers: Column headers
        title: Optional table title
        tablefmt: Table format (grid, simple, etc.)
    """
    if title:
        print(f"\n{title}")
        print("-" * len(title))

    print(tabulate(data, headers=headers, tablefmt=tablefmt))
    print()


def validate_solution(
    instance: ProblemInstance,
    solution: Solution,
    tolerance: float = 1e-6
) -> Tuple[bool, List[str]]:
    """
    Validate solution for feasibility.

    Args:
        instance: Problem instance
        solution: Solution to validate
        tolerance: Numerical tolerance

    Returns:
        Tuple of (is_valid, list_of_violations)
    """
    violations = []

    # Check time windows
    for aircraft in instance.aircraft:
        landing_time = solution.get_landing_time(aircraft.id)

        if landing_time < aircraft.appearance_time - tolerance:
            violations.append(
                f"Aircraft {aircraft.id}: Lands before appearance time "
                f"({landing_time:.2f} < {aircraft.appearance_time:.2f})"
            )

        if landing_time > aircraft.latest_time + tolerance:
            violations.append(
                f"Aircraft {aircraft.id}: Lands after latest time "
                f"({landing_time:.2f} > {aircraft.latest_time:.2f})"
            )

    # Check separations
    schedule_by_runway = solution.get_schedule_by_runway()

    for runway, landings in schedule_by_runway.items():
        for i in range(len(landings) - 1):
            aircraft_i_id, time_i = landings[i]
            aircraft_j_id, time_j = landings[i + 1]

            # Get indices
            idx_i = next(idx for idx, a in enumerate(instance.aircraft)
                        if a.id == aircraft_i_id)
            idx_j = next(idx for idx, a in enumerate(instance.aircraft)
                        if a.id == aircraft_j_id)

            required_sep = instance.get_separation(idx_i, idx_j)
            actual_sep = time_j - time_i

            if actual_sep < required_sep - tolerance:
                violations.append(
                    f"Runway {runway}: Insufficient separation between "
                    f"A{aircraft_i_id} and A{aircraft_j_id} "
                    f"({actual_sep:.2f} < {required_sep:.2f})"
                )

    is_valid = len(violations) == 0
    return is_valid, violations


def compare_solutions(
    instance: ProblemInstance,
    solutions: Dict[str, Solution],
    metric: str = 'cost'
) -> pd.DataFrame:
    """
    Compare multiple solutions.

    Args:
        instance: Problem instance
        solutions: Dictionary mapping name to Solution
        metric: Comparison metric ('cost', 'time', 'both')

    Returns:
        DataFrame with comparison
    """
    data = []

    for name, solution in solutions.items():
        row = {
            'Method': name,
            'Cost': solution.objective_value,
            'Time (s)': solution.solve_time,
            'Status': solution.status
        }

        # Add summary statistics
        summary = create_solution_summary(instance, solution)
        row['Early'] = summary['early_count']
        row['On-time'] = summary['on_time_count']
        row['Late'] = summary['late_count']

        data.append(row)

    df = pd.DataFrame(data)

    # Sort by metric
    if metric == 'cost':
        df = df.sort_values('Cost')
    elif metric == 'time':
        df = df.sort_values('Time (s)')

    return df


if __name__ == "__main__":
    # Test utilities
    from .data_loader import DataLoader
    from .heuristic import GreedyHeuristic

    print("Testing Utility Functions...")

    # Create sample instance and solution
    instance = DataLoader.create_sample_instance(num_aircraft=5)
    heuristic = GreedyHeuristic(instance, num_runways=1)
    solution = heuristic.solve()

    # Test functions
    print("\n1. Solution Summary:")
    summary = create_solution_summary(instance, solution)
    print(json.dumps(summary, indent=2, default=str))

    print("\n2. Detailed Solution:")
    print_solution_details(instance, solution, verbose=True)

    print("\n3. Solution Validation:")
    is_valid, violations = validate_solution(instance, solution)
    print(f"Valid: {is_valid}")
    if violations:
        print("Violations:")
        for v in violations:
            print(f"  - {v}")

    print("\nUtility functions test completed!")
