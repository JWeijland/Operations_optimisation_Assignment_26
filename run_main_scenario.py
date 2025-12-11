#!/usr/bin/env python3
"""
Main Scenario Runner - 10 Replications with Random Target Times
Consistent aircraft ratio, random target times per run
Peak hour: 11:00-11:40 (configurable percentage)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Setup paths
sys.path.insert(0, str(Path(__file__).parent / "aircraft_landing_scheduling"))

from code.data_loader import ProblemInstance, Aircraft, DataLoader
from code.solver import OptimalSolver
from code.visualization import ResultVisualizer
from code.utils import export_detailed_solution_table, export_detailed_solution_excel


class MainScenarioGenerator:
    """Generate main scenario with random target times."""

    def __init__(
        self,
        num_aircraft: int = 49,
        num_runways: int = 3,
        peak_percentage: float = 0.50,
        random_seed: int = None
    ):
        """
        Initialize main scenario generator.

        Args:
            num_aircraft: Number of aircraft (default 49 to maintain ratio)
            num_runways: Number of runways (default 3)
            peak_percentage: Percentage of aircraft in peak hour (default 0.50 = 50%)
            random_seed: Random seed for reproducibility
        """
        self.num_aircraft = num_aircraft
        self.num_runways = num_runways
        self.peak_percentage = peak_percentage
        self.random_seed = random_seed

        # Fixed ICAO ratio
        self.category_distribution = {
            'CAT-B': 0.10,
            'CAT-C': 0.30,
            'CAT-D': 0.45,
            'CAT-E': 0.12,
            'CAT-F': 0.03,
        }

        # Aircraft fleet (same as schiphol_realistic_peak.py)
        self.aircraft_fleet = {
            'CAT-B': [('B747-400', 20), ('B777-300ER', 40)],
            'CAT-C': [('B787-9', 16), ('A330-300', 29), ('B767-300', 30), ('A300-600', 15)],
            'CAT-D': [('B737-800', 226), ('A320-200', 95), ('A321-200', 40)],
            'CAT-E': [('E190', 100), ('E175', 60), ('ATR-72', 30)],
            'CAT-F': [('Cessna Citation', 20), ('Learjet 45', 15), ('ERJ-135', 10)],
        }

        # Cost structure
        self.cost_structure = {
            'CAT-B': {'early': 120, 'late': 250},
            'CAT-C': {'early': 100, 'late': 200},
            'CAT-D': {'early': 60, 'late': 150},
            'CAT-E': {'early': 40, 'late': 100},
            'CAT-F': {'early': 30, 'late': 80},
        }

        # Separation matrix (in minutes)
        self.separation_matrix_base = {
            'CAT-B': {'CAT-B': 1.0, 'CAT-C': 1.0, 'CAT-D': 100/60, 'CAT-E': 120/60, 'CAT-F': 140/60},
            'CAT-C': {'CAT-B': 1.0, 'CAT-C': 1.0, 'CAT-D': 80/60, 'CAT-E': 100/60, 'CAT-F': 120/60},
            'CAT-D': {'CAT-B': 1.0, 'CAT-C': 1.0, 'CAT-D': 1.0, 'CAT-E': 1.0, 'CAT-F': 120/60},
            'CAT-E': {'CAT-B': 1.0, 'CAT-C': 1.0, 'CAT-D': 1.0, 'CAT-E': 1.0, 'CAT-F': 100/60},
            'CAT-F': {'CAT-B': 1.0, 'CAT-C': 1.0, 'CAT-D': 1.0, 'CAT-E': 1.0, 'CAT-F': 80/60},
        }

    def generate_instance(self, replication_seed: int) -> ProblemInstance:
        """
        Generate a problem instance with random target times.

        Args:
            replication_seed: Seed for this specific replication

        Returns:
            ProblemInstance with random target times
        """
        np.random.seed(replication_seed)

        aircraft_list = []
        aircraft_id = 1

        # Generate aircraft with fixed ratio
        for category, ratio in self.category_distribution.items():
            n_category = int(self.num_aircraft * ratio)

            for _ in range(n_category):
                # Select random aircraft type from this category
                aircraft_types = self.aircraft_fleet[category]
                weights = [freq for _, freq in aircraft_types]
                aircraft_type = np.random.choice(
                    [name for name, _ in aircraft_types],
                    p=np.array(weights) / sum(weights)
                )

                # Generate RANDOM target time
                # Peak hour: 11:00-11:40 = 0-40 minutes
                # Off-peak: 11:40-13:00 = 40-120 minutes
                if np.random.random() < self.peak_percentage:
                    # Peak hour (first 40 minutes)
                    target_time = np.random.uniform(0, 40)
                else:
                    # Off-peak (40-120 minutes)
                    target_time = np.random.uniform(40, 120)

                # Time window: earliest = target - 5, latest = target + 5
                # With clipping to [0, 120]
                earliest = max(0, target_time - 5)
                latest = min(120, target_time + 5)

                # Costs
                early_cost = self.cost_structure[category]['early']
                late_cost = self.cost_structure[category]['late']

                # Create aircraft object
                aircraft = Aircraft(
                    id=aircraft_id,
                    appearance_time=earliest,
                    target_time=target_time,
                    latest_time=latest,
                    early_penalty=early_cost,
                    late_penalty=late_cost
                )

                aircraft_list.append(aircraft)
                aircraft_id += 1

        # Create separation matrix
        n = len(aircraft_list)
        separation_matrix = np.zeros((n, n))

        # Map aircraft to categories
        aircraft_categories = []
        for aircraft in aircraft_list:
            # Determine category based on penalty (reverse lookup)
            for cat, costs in self.cost_structure.items():
                if costs['early'] == aircraft.early_penalty and costs['late'] == aircraft.late_penalty:
                    aircraft_categories.append(cat)
                    break

        # Fill separation matrix
        for i in range(n):
            for j in range(n):
                leader_cat = aircraft_categories[i]
                follower_cat = aircraft_categories[j]
                separation_matrix[i, j] = self.separation_matrix_base[leader_cat][follower_cat]

        # Create problem instance
        instance = ProblemInstance(
            aircraft=aircraft_list,
            separation_matrix=separation_matrix,
            freeze_time=0.0
        )

        return instance


def run_main_scenario(
    num_replications: int = 10,
    num_aircraft: int = 49,
    num_runways: int = 3,
    peak_percentage: float = 0.50,
    base_seed: int = 1000,
    output_dir: str = "results/main_scenario"
):
    """
    Run main scenario with multiple replications.

    Args:
        num_replications: Number of replications (default 10)
        num_aircraft: Number of aircraft (default 49)
        num_runways: Number of runways (default 3)
        peak_percentage: Percentage in peak hour (default 0.50)
        base_seed: Base random seed (default 1000)
        output_dir: Output directory for results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("MAIN SCENARIO - MULTIPLE REPLICATIONS WITH RANDOM TARGET TIMES")
    print("="*80)
    print(f"Replications:     {num_replications}")
    print(f"Aircraft:         {num_aircraft}")
    print(f"Runways:          {num_runways}")
    print(f"Peak percentage:  {peak_percentage*100:.0f}% (11:00-11:40)")
    print(f"Time window:      11:00-13:00 (120 minutes)")
    print(f"Solver settings:  No time limit, 2% optimality gap")
    print(f"Output:           {output_dir}")
    print("="*80 + "\n")

    # Initialize generator
    generator = MainScenarioGenerator(
        num_aircraft=num_aircraft,
        num_runways=num_runways,
        peak_percentage=peak_percentage
    )

    # Storage for results
    results = []

    # Run replications
    for rep in range(1, num_replications + 1):
        print(f"\n{'='*80}")
        print(f"REPLICATION {rep}/{num_replications}")
        print(f"{'='*80}")

        # Generate instance with unique seed
        rep_seed = base_seed + rep
        instance = generator.generate_instance(rep_seed)

        print(f"Generated instance with seed {rep_seed}")
        print(f"  Aircraft: {instance.num_aircraft}")
        print(f"  Peak aircraft: {sum(1 for a in instance.aircraft if a.target_time <= 40)}")

        # Create solver
        solver = OptimalSolver(instance, instance_name=f"main_scenario_rep{rep}")

        # Solve
        # Note: time_limit=999999 = effectively no limit, gap=0.02 (default in solver)
        result = solver.solve_and_compare(
            num_runways=num_runways,
            time_limit=999999,  # Effectively no time limit (very high value)
            gap=0.02,  # 2% optimality gap
            use_multi_start=False
        )

        # Store results
        results.append({
            'replication': rep,
            'seed': rep_seed,
            'heuristic_cost': result.heuristic_cost,
            'optimal_cost': result.optimal_cost,
            'gap_percent': result.gap,
            'heuristic_time': result.heuristic_time,
            'optimal_time': result.optimal_time,
            'optimal_status': result.optimal_solution.status if result.optimal_solution else 'Not Solved'
        })

        # Generate outputs for this replication
        rep_output_dir = output_path / f"replication_{rep}"
        rep_output_dir.mkdir(exist_ok=True)

        print(f"\nGenerating outputs for replication {rep}...")

        # Visualizations
        viz = ResultVisualizer(output_dir=str(rep_output_dir))

        viz.plot_gantt_chart(
            instance,
            result.heuristic_solution,
            title=f"Main Scenario Rep {rep} - Heuristic Solution",
            filename=f"rep{rep}_heuristic_gantt"
        )

        viz.plot_gantt_chart(
            instance,
            result.optimal_solution,
            title=f"Main Scenario Rep {rep} - Optimal Solution",
            filename=f"rep{rep}_optimal_gantt"
        )

        viz.plot_comparison(
            instance,
            result.heuristic_solution,
            result.optimal_solution,
            filename=f"rep{rep}_comparison"
        )

        viz.plot_cost_breakdown(
            instance,
            result.optimal_solution,
            filename=f"rep{rep}_cost_breakdown"
        )

        # Export Excel tables
        tables_dir = rep_output_dir / "tables"
        tables_dir.mkdir(exist_ok=True)

        export_detailed_solution_excel(
            instance,
            result.heuristic_solution,
            str(tables_dir / f"rep{rep}_heuristic_detailed.xlsx"),
            "Heuristic"
        )

        export_detailed_solution_excel(
            instance,
            result.optimal_solution,
            str(tables_dir / f"rep{rep}_optimal_detailed.xlsx"),
            "Optimal"
        )

        print(f"✓ Outputs saved to {rep_output_dir}")

    # Calculate statistics
    print(f"\n{'='*80}")
    print("STATISTICS OVER ALL REPLICATIONS")
    print(f"{'='*80}\n")

    df_results = pd.DataFrame(results)

    # Summary statistics
    stats = {
        'Heuristic Cost': {
            'Mean': df_results['heuristic_cost'].mean(),
            'Std': df_results['heuristic_cost'].std(),
            'Min': df_results['heuristic_cost'].min(),
            'Max': df_results['heuristic_cost'].max(),
        },
        'Optimal Cost': {
            'Mean': df_results['optimal_cost'].mean(),
            'Std': df_results['optimal_cost'].std(),
            'Min': df_results['optimal_cost'].min(),
            'Max': df_results['optimal_cost'].max(),
        },
        'Gap (%)': {
            'Mean': df_results['gap_percent'].mean(),
            'Std': df_results['gap_percent'].std(),
            'Min': df_results['gap_percent'].min(),
            'Max': df_results['gap_percent'].max(),
        }
    }

    # Print statistics
    print(f"{'Metric':<20} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
    print("-"*80)
    for metric, values in stats.items():
        print(f"{metric:<20} {values['Mean']:<15.2f} {values['Std']:<15.2f} "
              f"{values['Min']:<15.2f} {values['Max']:<15.2f}")

    # Save results to CSV
    results_file = output_path / "replication_results.csv"
    df_results.to_csv(results_file, index=False)
    print(f"\n✓ Results saved to {results_file}")

    # Save statistics
    stats_file = output_path / "statistics_summary.txt"
    with open(stats_file, 'w') as f:
        f.write("MAIN SCENARIO - STATISTICS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Replications:     {num_replications}\n")
        f.write(f"Aircraft:         {num_aircraft}\n")
        f.write(f"Runways:          {num_runways}\n")
        f.write(f"Peak percentage:  {peak_percentage*100:.0f}%\n\n")
        f.write(f"{'Metric':<20} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}\n")
        f.write("-"*80 + "\n")
        for metric, values in stats.items():
            f.write(f"{metric:<20} {values['Mean']:<15.2f} {values['Std']:<15.2f} "
                   f"{values['Min']:<15.2f} {values['Max']:<15.2f}\n")

    print(f"✓ Statistics saved to {stats_file}")

    print(f"\n{'='*80}")
    print("MAIN SCENARIO COMPLETED")
    print(f"{'='*80}")
    print(f"Results directory: {output_dir}")
    print(f"  - {num_replications} replication folders")
    print(f"  - replication_results.csv (detailed results)")
    print(f"  - statistics_summary.txt (summary)")
    print(f"{'='*80}\n")

    return df_results, stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Main Scenario with Multiple Replications")
    parser.add_argument('--replications', type=int, default=10, help='Number of replications (default: 10)')
    parser.add_argument('--aircraft', type=int, default=49, help='Number of aircraft (default: 49)')
    parser.add_argument('--runways', type=int, default=3, help='Number of runways (default: 3)')
    parser.add_argument('--peak-pct', type=float, default=0.50, help='Peak percentage (default: 0.50)')
    parser.add_argument('--seed', type=int, default=1000, help='Base random seed (default: 1000)')

    args = parser.parse_args()

    run_main_scenario(
        num_replications=args.replications,
        num_aircraft=args.aircraft,
        num_runways=args.runways,
        peak_percentage=args.peak_pct,
        base_seed=args.seed
    )
