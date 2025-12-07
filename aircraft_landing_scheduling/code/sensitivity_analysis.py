#!/usr/bin/env python3
"""
Sensitivity Analysis Module
Generates scenarios and runs experiments to analyze impact of aircraft count and runway count
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import time
from datetime import datetime

try:
    from .sensitivity_config import SensitivityAnalysisConfig
    from .schiphol_scenarios import create_schiphol_evening_rush
    from .solver import OptimalSolver
    from .data_loader import ProblemInstance
except ImportError:
    # Standalone mode - use absolute imports
    import sensitivity_config
    import schiphol_scenarios
    import solver as solver_module
    import data_loader

    SensitivityAnalysisConfig = sensitivity_config.SensitivityAnalysisConfig
    create_schiphol_evening_rush = schiphol_scenarios.create_schiphol_evening_rush
    OptimalSolver = solver_module.OptimalSolver
    ProblemInstance = data_loader.ProblemInstance


class SensitivityAnalyzer:
    """
    Runs sensitivity analysis experiments across different aircraft and runway counts.
    """

    def __init__(self, config: SensitivityAnalysisConfig):
        """
        Initialize sensitivity analyzer.

        Args:
            config: Configuration object with experiment parameters
        """
        self.config = config
        self.results = []
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_full_analysis(self) -> pd.DataFrame:
        """
        Run complete sensitivity analysis across all aircraft/runway combinations.

        Returns:
            DataFrame with all results
        """
        print("\n" + "=" * 70)
        print("STARTING SENSITIVITY ANALYSIS")
        print("=" * 70)
        self.config.print_summary()

        start_time = time.time()
        total_scenarios = len(self.config.aircraft_counts) * len(self.config.runway_counts) * self.config.num_replications
        scenario_counter = 0

        # Run experiments for each combination
        for num_aircraft in self.config.aircraft_counts:
            for num_runways in self.config.runway_counts:
                for replication in range(self.config.num_replications):
                    scenario_counter += 1
                    print(f"\n{'=' * 70}")
                    print(f"Scenario {scenario_counter}/{total_scenarios}")
                    print(f"Aircraft: {num_aircraft}, Runways: {num_runways}, Rep: {replication + 1}/{self.config.num_replications}")
                    print(f"{'=' * 70}")

                    # Generate and solve scenario
                    result = self._run_single_scenario(
                        num_aircraft=num_aircraft,
                        num_runways=num_runways,
                        replication=replication
                    )

                    if result:
                        self.results.append(result)

        # Create results DataFrame
        df = pd.DataFrame(self.results)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"sensitivity_results_{timestamp}.csv"
        df.to_csv(results_file, index=False)

        total_time = time.time() - start_time
        print(f"\n{'=' * 70}")
        print(f"SENSITIVITY ANALYSIS COMPLETED")
        print(f"{'=' * 70}")
        print(f"Total scenarios: {len(self.results)}")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Results saved to: {results_file}")
        print(f"{'=' * 70}\n")

        return df

    def _run_single_scenario(
        self,
        num_aircraft: int,
        num_runways: int,
        replication: int
    ) -> Dict:
        """
        Generate and solve a single scenario.

        Args:
            num_aircraft: Number of aircraft
            num_runways: Number of runways
            replication: Replication number

        Returns:
            Dictionary with results
        """
        try:
            # Generate unique seed for this scenario
            scenario_seed = self.config.random_seed + (num_aircraft * 1000) + (num_runways * 100) + replication

            # Generate scenario name
            scenario_name = f"sensitivity_a{num_aircraft}_r{num_runways}_rep{replication}"

            print(f"\nGenerating scenario: {scenario_name}")
            print(f"  Rush hour: {self.config.rush_hour.start_minute:.0f}-{self.config.rush_hour.end_minute:.0f} min")
            print(f"  Rush probability: {self.config.rush_hour.probability * 100:.0f}%")

            # Generate problem instance using create_schiphol_evening_rush
            instance = create_schiphol_evening_rush(
                num_aircraft=num_aircraft,
                num_runways=num_runways,
                scenario_name=scenario_name,
                heavy_ratio=self.config.aircraft_mix.heavy_ratio,
                medium_ratio=self.config.aircraft_mix.medium_ratio,
                light_ratio=self.config.aircraft_mix.light_ratio,
                seed=scenario_seed,
                peak_hour_probability=self.config.rush_hour.probability
            )

            # Solve the instance
            solver = OptimalSolver(instance, instance_name=scenario_name)

            # Solve with both heuristic and optimal
            comparison_result = solver.solve_and_compare(
                num_runways=num_runways,
                time_limit=self.config.time_limit,
                use_multi_start=False
            )

            # Extract results
            result = {
                'num_aircraft': num_aircraft,
                'num_runways': num_runways,
                'replication': replication,
                'scenario_name': scenario_name,
                'rush_hour_prob': self.config.rush_hour.probability,
                'heuristic_cost': comparison_result.heuristic_cost,
                'optimal_cost': comparison_result.optimal_cost,
                'gap_percent': comparison_result.gap,
                'heuristic_time_s': comparison_result.heuristic_time,
                'optimal_time_s': comparison_result.optimal_time,
                'speedup': comparison_result.speedup,
                'heavy_ratio': self.config.aircraft_mix.heavy_ratio,
                'medium_ratio': self.config.aircraft_mix.medium_ratio,
                'light_ratio': self.config.aircraft_mix.light_ratio,
                'seed': scenario_seed
            }

            # Save detailed results if requested
            if self.config.save_detailed_results:
                self._save_detailed_solution(
                    scenario_name=scenario_name,
                    instance=instance,
                    comparison_result=comparison_result
                )

            print(f"  ✓ Completed: Optimal cost = {comparison_result.optimal_cost:.2f}, "
                  f"Gap = {comparison_result.gap:.2f}%")

            return result

        except Exception as e:
            print(f"  ✗ Error in scenario: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_detailed_solution(self, scenario_name: str, instance: ProblemInstance, comparison_result):
        """Save detailed solution for a scenario."""
        try:
            from .utils import export_detailed_solution_table
        except ImportError:
            import utils
            export_detailed_solution_table = utils.export_detailed_solution_table

        detailed_dir = self.output_dir / "detailed_solutions"
        detailed_dir.mkdir(parents=True, exist_ok=True)

        # Save heuristic solution
        heuristic_file = detailed_dir / f"{scenario_name}_heuristic.csv"
        export_detailed_solution_table(
            instance,
            comparison_result.heuristic_solution,
            str(heuristic_file),
            "Heuristic"
        )

        # Save optimal solution
        optimal_file = detailed_dir / f"{scenario_name}_optimal.csv"
        export_detailed_solution_table(
            instance,
            comparison_result.optimal_solution,
            str(optimal_file),
            "Optimal"
        )

    def aggregate_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate results across replications.

        Args:
            df: DataFrame with all results

        Returns:
            DataFrame with aggregated statistics
        """
        if self.config.num_replications == 1:
            # No aggregation needed
            return df

        # Group by aircraft and runway counts
        grouped = df.groupby(['num_aircraft', 'num_runways'])

        # Calculate statistics
        agg_df = grouped.agg({
            'optimal_cost': ['mean', 'std', 'min', 'max'],
            'heuristic_cost': ['mean', 'std', 'min', 'max'],
            'gap_percent': ['mean', 'std'],
            'optimal_time_s': ['mean', 'std'],
            'heuristic_time_s': ['mean', 'std'],
            'speedup': ['mean', 'std']
        }).reset_index()

        # Flatten column names
        agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]

        return agg_df

    def create_pivot_table(self, df: pd.DataFrame, metric: str = 'optimal_cost') -> pd.DataFrame:
        """
        Create pivot table for heatmap visualization.

        Args:
            df: Results DataFrame
            metric: Metric to pivot (e.g., 'optimal_cost', 'gap_percent')

        Returns:
            Pivot table with aircraft as rows, runways as columns
        """
        # If multiple replications, use mean
        if self.config.num_replications > 1:
            pivot_data = df.groupby(['num_aircraft', 'num_runways'])[metric].mean().reset_index()
        else:
            pivot_data = df[['num_aircraft', 'num_runways', metric]]

        # Create pivot table
        pivot = pivot_data.pivot(
            index='num_aircraft',
            columns='num_runways',
            values=metric
        )

        return pivot

    def print_summary_statistics(self, df: pd.DataFrame):
        """Print summary statistics of the analysis."""
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70)

        print(f"\nTotal scenarios run: {len(df)}")

        print(f"\nOptimal Cost Statistics:")
        print(f"  Mean: {df['optimal_cost'].mean():.2f}")
        print(f"  Std: {df['optimal_cost'].std():.2f}")
        print(f"  Min: {df['optimal_cost'].min():.2f}")
        print(f"  Max: {df['optimal_cost'].max():.2f}")

        print(f"\nGap Statistics:")
        print(f"  Mean gap: {df['gap_percent'].mean():.2f}%")
        print(f"  Std gap: {df['gap_percent'].std():.2f}%")
        print(f"  Max gap: {df['gap_percent'].max():.2f}%")

        print(f"\nSolve Time Statistics:")
        print(f"  Mean optimal time: {df['optimal_time_s'].mean():.2f}s")
        print(f"  Mean heuristic time: {df['heuristic_time_s'].mean():.4f}s")
        print(f"  Mean speedup: {df['speedup'].mean():.1f}x")

        print("\nCost by Aircraft Count:")
        cost_by_aircraft = df.groupby('num_aircraft')['optimal_cost'].mean()
        for aircraft, cost in cost_by_aircraft.items():
            print(f"  {aircraft} aircraft: {cost:.2f}")

        print("\nCost by Runway Count:")
        cost_by_runway = df.groupby('num_runways')['optimal_cost'].mean()
        for runway, cost in cost_by_runway.items():
            print(f"  {runway} runway(s): {cost:.2f}")

        print("=" * 70)


def run_sensitivity_from_config(config_file: str = None, config_obj: SensitivityAnalysisConfig = None):
    """
    Run sensitivity analysis from configuration file or object.

    Args:
        config_file: Path to YAML configuration file
        config_obj: Configuration object (alternative to config_file)
    """
    # Load configuration
    if config_file:
        config = SensitivityAnalysisConfig.load_from_yaml(config_file)
    elif config_obj:
        config = config_obj
    else:
        raise ValueError("Must provide either config_file or config_obj")

    # Create analyzer
    analyzer = SensitivityAnalyzer(config)

    # Run analysis
    results_df = analyzer.run_full_analysis()

    # Print summary statistics
    analyzer.print_summary_statistics(results_df)

    # Generate visualizations if requested
    if config.generate_heatmaps:
        from sensitivity_visualization import create_all_heatmaps
        create_all_heatmaps(results_df, str(analyzer.output_dir))

    return results_df, analyzer


if __name__ == "__main__":
    # Example: Run with default configuration
    try:
        from .sensitivity_config import create_default_config
    except ImportError:
        from sensitivity_config import create_default_config

    config = create_default_config()
    results_df, analyzer = run_sensitivity_from_config(config_obj=config)

    print("\n✓ Sensitivity analysis completed!")
