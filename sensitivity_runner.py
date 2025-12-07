#!/usr/bin/env python3
"""
Simplified Sensitivity Analysis Runner
Works directly with existing codebase structure
"""

import sys
from pathlib import Path

# Setup paths to use existing code
sys.path.insert(0, str(Path(__file__).parent / "aircraft_landing_scheduling"))

from code.sensitivity_config import (
    SensitivityAnalysisConfig,
    RushHourConfig,
    AircraftMixConfig
)
from code.schiphol_scenarios import create_schiphol_evening_rush
from code.solver import OptimalSolver
from code.data_loader import ProblemInstance
from code.utils import export_detailed_solution_table

import pandas as pd
import numpy as np
import time
from datetime import datetime
import argparse


def run_sensitivity_analysis(config: SensitivityAnalysisConfig):
    """Run complete sensitivity analysis."""
    print("\n" + "=" * 70)
    print("STARTING SENSITIVITY ANALYSIS")
    print("=" * 70)
    config.print_summary()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    start_time = time.time()
    total_scenarios = len(config.aircraft_counts) * len(config.runway_counts) * config.num_replications
    scenario_counter = 0

    # Run all scenarios
    for num_aircraft in config.aircraft_counts:
        for num_runways in config.runway_counts:
            for replication in range(config.num_replications):
                scenario_counter += 1
                print(f"\n{'=' * 70}")
                print(f"Scenario {scenario_counter}/{total_scenarios}")
                print(f"Aircraft: {num_aircraft}, Runways: {num_runways}, Rep: {replication + 1}/{config.num_replications}")
                print(f"{'=' * 70}")

                # Generate scenario
                scenario_seed = config.random_seed + (num_aircraft * 1000) + (num_runways * 100) + replication
                scenario_name = f"sensitivity_a{num_aircraft}_r{num_runways}_rep{replication}"

                print(f"\nGenerating scenario: {scenario_name}")
                instance = create_schiphol_evening_rush(
                    num_aircraft=num_aircraft,
                    num_runways=num_runways,
                    scenario_name=scenario_name,
                    heavy_ratio=config.aircraft_mix.heavy_ratio,
                    medium_ratio=config.aircraft_mix.medium_ratio,
                    light_ratio=config.aircraft_mix.light_ratio,
                    seed=scenario_seed,
                    peak_hour_probability=config.rush_hour.probability
                )

                # Solve
                solver = OptimalSolver(instance, instance_name=scenario_name)
                comparison_result = solver.solve_and_compare(
                    num_runways=num_runways,
                    time_limit=config.time_limit,
                    use_multi_start=False
                )

                # Store results
                result = {
                    'num_aircraft': num_aircraft,
                    'num_runways': num_runways,
                    'replication': replication,
                    'scenario_name': scenario_name,
                    'rush_hour_prob': config.rush_hour.probability,
                    'heuristic_cost': comparison_result.heuristic_cost,
                    'optimal_cost': comparison_result.optimal_cost,
                    'gap_percent': comparison_result.gap,
                    'heuristic_time_s': comparison_result.heuristic_time,
                    'optimal_time_s': comparison_result.optimal_time,
                    'speedup': comparison_result.speedup
                }
                results.append(result)

                print(f"  ✓ Optimal cost = {comparison_result.optimal_cost:.2f}, Gap = {comparison_result.gap:.2f}%")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"sensitivity_results_{timestamp}.csv"
    df.to_csv(results_file, index=False)

    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"ANALYSIS COMPLETED")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Results saved to: {results_file}")
    print(f"{'=' * 70}\n")

    # Print summary
    print_summary(df)

    # Generate visualizations
    if config.generate_heatmaps:
        generate_heatmaps(df, str(output_dir))

    return df


def print_summary(df):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"\nTotal scenarios: {len(df)}")
    print(f"\nOptimal Cost:")
    print(f"  Mean: {df['optimal_cost'].mean():.2f}")
    print(f"  Min: {df['optimal_cost'].min():.2f}")
    print(f"  Max: {df['optimal_cost'].max():.2f}")
    print(f"\nGap:")
    print(f"  Mean: {df['gap_percent'].mean():.2f}%")
    print(f"  Max: {df['gap_percent'].max():.2f}%")
    print("=" * 70)


def generate_heatmaps(df, output_dir):
    """Generate heatmap visualizations."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("\nGenerating heatmaps...")
    heatmap_dir = Path(output_dir) / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate replications first (take mean)
    df_agg = df.groupby(['num_aircraft', 'num_runways']).agg({
        'optimal_cost': 'mean',
        'gap_percent': 'mean',
        'optimal_time_s': 'mean'
    }).reset_index()

    # Optimal cost heatmap
    pivot = df_agg.pivot(index='num_aircraft', columns='num_runways', values='optimal_cost')

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', linewidths=0.5)
    plt.title('Optimal Landing Cost by Aircraft and Runway Count', fontweight='bold', fontsize=14)
    plt.xlabel('Number of Runways', fontweight='bold')
    plt.ylabel('Number of Aircraft', fontweight='bold')
    plt.tight_layout()
    plt.savefig(heatmap_dir / 'optimal_cost_heatmap.png', dpi=300)
    print(f"  ✓ Saved: optimal_cost_heatmap.png")
    plt.close()

    # Gap heatmap
    pivot_gap = df_agg.pivot(index='num_aircraft', columns='num_runways', values='gap_percent')

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_gap, annot=True, fmt='.2f', cmap='RdYlGn_r', linewidths=0.5)
    plt.title('Heuristic Gap (%) by Aircraft and Runway Count', fontweight='bold', fontsize=14)
    plt.xlabel('Number of Runways', fontweight='bold')
    plt.ylabel('Number of Aircraft', fontweight='bold')
    plt.tight_layout()
    plt.savefig(heatmap_dir / 'gap_percent_heatmap.png', dpi=300)
    print(f"  ✓ Saved: gap_percent_heatmap.png")
    plt.close()

    print(f"\n✓ Heatmaps saved to: {heatmap_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run Sensitivity Analysis")
    parser.add_argument('--quick-test', action='store_true', help='Quick 2x2 test')
    parser.add_argument('--aircraft', type=int, nargs='+', default=[20, 30, 40, 50])
    parser.add_argument('--runways', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--rush-prob', type=float, default=0.5)
    parser.add_argument('--time-limit', type=int, default=50)
    parser.add_argument('--name', type=str, default='sensitivity_analysis')

    args = parser.parse_args()

    # Create configuration
    if args.quick_test:
        aircraft_counts = [10, 20]
        runway_counts = [1, 2]
        time_limit = 60
        experiment_name = "quick_test"
    else:
        aircraft_counts = args.aircraft
        runway_counts = args.runways
        time_limit = args.time_limit
        experiment_name = args.name

    config = SensitivityAnalysisConfig(
        experiment_name=experiment_name,
        aircraft_counts=aircraft_counts,
        runway_counts=runway_counts,
        rush_hour=RushHourConfig(
            start_minute=60.0,
            end_minute=90.0,
            probability=args.rush_prob
        ),
        time_limit=time_limit,
        output_dir=f"results/{experiment_name}"
    )

    # Run analysis
    try:
        results_df = run_sensitivity_analysis(config)
        print("\n✓ Sensitivity analysis completed successfully!")
    except KeyboardInterrupt:
        print("\n\n✗ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
