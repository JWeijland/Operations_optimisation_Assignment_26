#!/usr/bin/env python3
"""
COMPLETE SENSITIVITY ANALYSIS RUNNER
=====================================
This script runs the complete sensitivity analysis for:
- Aircraft: 10, 20, 30, 40, 50
- Runways: 1, 2, 3
- Time limit: 300 seconds per instance

It generates:
1. A comprehensive table with all individual runs (solve time, variables, constraints, gap, status)
2. Heatmaps for all key metrics
3. Summary statistics

USAGE:
    python run_complete_sensitivity.py

Then wait... it will take a while (approximately 45 scenarios × 300s = ~3.75 hours max)
"""

import sys
from pathlib import Path
import pandas as pd
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.sensitivity_config import SensitivityAnalysisConfig, RushHourConfig, AircraftMixConfig
from code.sensitivity_analysis import SensitivityAnalyzer
from code.sensitivity_visualization import create_all_heatmaps


def create_comprehensive_results_table(results_df: pd.DataFrame, output_dir: Path):
    """
    Create the comprehensive results table with ALL details.

    Columns:
    - num_aircraft
    - num_runways
    - replication
    - optimal_cost
    - heuristic_cost
    - gap_percent
    - solve_time_s
    - heuristic_time_s
    - num_variables (estimated)
    - num_constraints (estimated)
    - status
    - scenario_name
    """
    print("\n" + "="*80)
    print("CREATING COMPREHENSIVE RESULTS TABLE")
    print("="*80)

    # Calculate number of variables and constraints
    # Based on the model formulation:
    # Variables per instance:
    #   - x (landing times): n variables
    #   - alpha (early): n variables
    #   - beta (late): n variables
    #   - delta (ordering): n*(n-1)/2 variables
    #   - For multiple runways:
    #       - y (runway assignment): n*r variables
    #       - z (same runway indicator): n*(n-1)/2 variables

    def estimate_variables(n_aircraft, n_runways):
        """Estimate number of decision variables."""
        n = n_aircraft
        r = n_runways

        # Basic variables
        x_vars = n  # landing times
        alpha_vars = n  # early deviation
        beta_vars = n  # late deviation
        delta_vars = n * (n - 1) // 2  # ordering

        if r == 1:
            return x_vars + alpha_vars + beta_vars + delta_vars
        else:
            y_vars = n * r  # runway assignments
            z_vars = n * (n - 1) // 2  # same runway indicators
            return x_vars + alpha_vars + beta_vars + delta_vars + y_vars + z_vars

    def estimate_constraints(n_aircraft, n_runways):
        """Estimate number of constraints."""
        n = n_aircraft
        r = n_runways

        # Basic constraints
        time_window = 2 * n  # earliest and latest bounds
        target_dev = n  # target deviation definition
        separation = 2 * n * (n - 1) // 2  # separation (2 per pair)

        if r == 1:
            return time_window + target_dev + separation
        else:
            runway_assign = n  # each aircraft to one runway
            same_runway = n * (n - 1) // 2 * r  # same runway linking
            return time_window + target_dev + separation + runway_assign + same_runway

    # Add calculated columns
    results_df['num_variables'] = results_df.apply(
        lambda row: estimate_variables(row['num_aircraft'], row['num_runways']), axis=1
    )
    results_df['num_constraints'] = results_df.apply(
        lambda row: estimate_constraints(row['num_aircraft'], row['num_runways']), axis=1
    )

    # Reorder columns for better readability
    column_order = [
        'num_aircraft',
        'num_runways',
        'replication',
        'num_variables',
        'num_constraints',
        'optimal_cost',
        'heuristic_cost',
        'gap_percent',
        'optimal_time_s',
        'heuristic_time_s',
        'speedup',
        'scenario_name',
        'seed'
    ]

    # Only include columns that exist
    column_order = [col for col in column_order if col in results_df.columns]
    comprehensive_df = results_df[column_order].copy()

    # Add status column (all should be completed)
    comprehensive_df['status'] = 'Completed'

    # Save comprehensive table
    output_file = output_dir / 'comprehensive_results_table.csv'
    comprehensive_df.to_csv(output_file, index=False)
    print(f"✓ Comprehensive table saved: {output_file}")

    # Also save as Excel for easier viewing
    excel_file = output_dir / 'comprehensive_results_table.xlsx'
    try:
        comprehensive_df.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"✓ Excel version saved: {excel_file}")
    except ImportError:
        print("  (Excel export skipped - install openpyxl for Excel support)")

    # Print summary
    print(f"\nTable Statistics:")
    print(f"  Total runs: {len(comprehensive_df)}")
    print(f"  Aircraft counts: {sorted(comprehensive_df['num_aircraft'].unique())}")
    print(f"  Runway counts: {sorted(comprehensive_df['num_runways'].unique())}")
    print(f"  Average solve time: {comprehensive_df['optimal_time_s'].mean():.1f}s")
    print(f"  Max solve time: {comprehensive_df['optimal_time_s'].max():.1f}s")
    print(f"  Average gap: {comprehensive_df['gap_percent'].mean():.2f}%")
    print(f"  Max gap: {comprehensive_df['gap_percent'].max():.2f}%")

    return comprehensive_df


def print_banner():
    """Print a nice banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║           AIRCRAFT LANDING SCHEDULING - COMPLETE SENSITIVITY ANALYSIS        ║
║                                                                              ║
║  This will run a comprehensive sensitivity analysis with:                   ║
║    • Aircraft counts: 10, 20, 30, 40, 50                                    ║
║    • Runway counts: 1, 2, 3                                                 ║
║    • Time limit: 300 seconds per scenario                                   ║
║    • Total scenarios: 15 combinations × 1 replication = 15 runs             ║
║                                                                              ║
║  Estimated time: 15-75 minutes (depending on solve times)                   ║
║                                                                              ║
║  Outputs:                                                                   ║
║    1. Comprehensive results table (CSV + Excel)                             ║
║    2. Heatmaps for all metrics                                              ║
║    3. Comparison plots                                                      ║
║    4. Summary statistics                                                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def main():
    """Main execution function."""
    print_banner()

    # Confirm execution
    print("\nThis analysis will take significant time to complete.")
    response = input("Do you want to proceed? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Analysis cancelled.")
        return

    # Create configuration
    print("\n" + "="*80)
    print("SETTING UP CONFIGURATION")
    print("="*80)

    config = SensitivityAnalysisConfig(
        experiment_name="complete_sensitivity_10_to_50_aircraft",
        aircraft_counts=[10, 20, 30, 40, 50],  # 5 aircraft counts
        runway_counts=[1, 2, 3],                 # 3 runway counts
        time_limit=300,                          # 5 minutes per scenario
        num_replications=1,                      # 1 replication (no randomness averaging)
        random_seed=42,
        rush_hour=RushHourConfig(
            start_minute=60.0,
            end_minute=90.0,
            probability=0.5  # 50% of aircraft in rush hour
        ),
        aircraft_mix=AircraftMixConfig(
            heavy_ratio=0.30,   # 30% heavy
            medium_ratio=0.67,  # 67% medium
            light_ratio=0.03    # 3% light
        ),
        output_dir="results/complete_sensitivity_analysis",
        save_detailed_results=False,  # Don't save individual solutions (saves space)
        generate_heatmaps=True
    )

    config.print_summary()

    # Create analyzer
    analyzer = SensitivityAnalyzer(config)

    # Run complete analysis
    print("\n" + "="*80)
    print("STARTING SENSITIVITY ANALYSIS")
    print("="*80)
    print("\nThis will take a while... Go get a coffee (or two)!\n")

    start_time = time.time()

    # Run analysis
    results_df = analyzer.run_full_analysis()

    total_time = time.time() - start_time

    # Create comprehensive results table
    comprehensive_df = create_comprehensive_results_table(
        results_df,
        Path(config.output_dir)
    )

    # Print summary statistics
    analyzer.print_summary_statistics(results_df)

    # Generate visualizations
    if config.generate_heatmaps:
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        create_all_heatmaps(results_df, config.output_dir, config.num_replications)

    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n✓ Total execution time: {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
    print(f"✓ Total scenarios completed: {len(results_df)}")
    print(f"✓ All results saved to: {config.output_dir}")
    print(f"\nKey outputs:")
    print(f"  • Comprehensive table: {config.output_dir}/comprehensive_results_table.csv")
    print(f"  • Heatmaps: {config.output_dir}/heatmaps/")
    print(f"  • Comparison plots: {config.output_dir}/plots/")
    print(f"  • Raw results: {config.output_dir}/")
    print("\n" + "="*80)
    print("\nYo Jelle, je hoeft alleen op deze knop te klikken,")
    print("super lang te wachten en dan krijg je het resultaat van")
    print("de sensitivity analysis in een heat map vorm en de")
    print("desbetreffende super grote tabel van alle individuele")
    print("runs waarin de solve time, variables, constraints, gap")
    print("en status allemaal erin staan!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
