#!/usr/bin/env python3
"""
Run Fixed Scenarios
Simple script to run the 4 predefined scenarios
"""

import sys
from pathlib import Path
import argparse

# Setup paths
sys.path.insert(0, str(Path(__file__).parent / "aircraft_landing_scheduling"))
sys.path.insert(0, str(Path(__file__).parent / "scenarios"))

from code.solver import OptimalSolver
from code.visualization import ResultVisualizer
from code.utils import export_detailed_solution_table, export_detailed_solution_excel
from scenario_definitions import SCENARIOS, get_scenario, print_scenarios


def run_scenario(scenario_id, time_limit=50, output_dir=None):
    """
    Run a single scenario.

    Args:
        scenario_id: Scenario ID (1-4)
        time_limit: Time limit for solver in seconds
        output_dir: Output directory (default: results/scenario_X)
    """
    # Get scenario
    instance, info = get_scenario(scenario_id)

    # Setup output directory
    if output_dir is None:
        output_dir = f"results/scenario_{scenario_id}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"RUNNING SCENARIO {scenario_id}")
    print("=" * 70)
    print(f"Name: {info['name']}")
    print(f"Aircraft: {info['num_aircraft']}")
    print(f"Runways: {info['num_runways']}")
    print(f"Description: {info['description']}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Create solver
    scenario_name = f"scenario_{scenario_id}"
    solver = OptimalSolver(instance, instance_name=scenario_name)

    # Solve with both methods
    result = solver.solve_and_compare(
        num_runways=info['num_runways'],
        time_limit=time_limit,
        use_multi_start=False
    )

    # Generate visualizations
    print("\nGenerating visualizations...")
    viz = ResultVisualizer(output_dir=output_dir)

    # Gantt charts
    viz.plot_gantt_chart(
        instance,
        result.heuristic_solution,
        title=f"{info['name']} - Heuristic Solution",
        filename=f"{scenario_name}_heuristic_gantt"
    )

    viz.plot_gantt_chart(
        instance,
        result.optimal_solution,
        title=f"{info['name']} - Optimal Solution",
        filename=f"{scenario_name}_optimal_gantt"
    )

    # Comparison plot
    viz.plot_comparison(
        instance,
        result.heuristic_solution,
        result.optimal_solution,
        filename=f"{scenario_name}_comparison"
    )

    # Cost breakdown
    viz.plot_cost_breakdown(
        instance,
        result.optimal_solution,
        filename=f"{scenario_name}_cost_breakdown"
    )

    # Export detailed tables
    print("\nExporting solution tables...")
    tables_dir = output_path / "tables"
    tables_dir.mkdir(exist_ok=True)

    export_detailed_solution_table(
        instance,
        result.heuristic_solution,
        str(tables_dir / f"{scenario_name}_heuristic.csv"),
        "Heuristic"
    )

    export_detailed_solution_table(
        instance,
        result.optimal_solution,
        str(tables_dir / f"{scenario_name}_optimal.csv"),
        "Optimal"
    )

    export_detailed_solution_excel(
        instance,
        result.heuristic_solution,
        str(tables_dir / f"{scenario_name}_heuristic.xlsx"),
        "Heuristic"
    )

    export_detailed_solution_excel(
        instance,
        result.optimal_solution,
        str(tables_dir / f"{scenario_name}_optimal.xlsx"),
        "Optimal"
    )

    print("\n" + "=" * 70)
    print(f"SCENARIO {scenario_id} COMPLETED")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")
    print("=" * 70 + "\n")

    return result


def run_all_scenarios(time_limit=50):
    """Run all 4 scenarios."""
    print("\n" + "=" * 70)
    print("RUNNING ALL SCENARIOS")
    print("=" * 70)

    results = {}
    for scenario_id in ['1', '2', '3', '4']:
        result = run_scenario(scenario_id, time_limit=time_limit)
        results[scenario_id] = result

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY - ALL SCENARIOS")
    print("=" * 70)

    for scenario_id, result in results.items():
        info = SCENARIOS[scenario_id]
        print(f"\nScenario {scenario_id}: {info['name']}")
        print(f"  Optimal Cost: €{result.optimal_cost:.2f}")
        print(f"  Heuristic Cost: €{result.heuristic_cost:.2f}")
        print(f"  Gap: {result.gap:.2f}%")
        print(f"  Solve Time: {result.optimal_time:.2f}s")

    print("=" * 70 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run Aircraft Landing Scheduling Scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single scenario
  python run_scenarios.py --scenario 1

  # Run all scenarios
  python run_scenarios.py --all

  # Run with custom time limit
  python run_scenarios.py --scenario 2 --time-limit 100

  # List available scenarios
  python run_scenarios.py --list
        """
    )

    parser.add_argument(
        '--scenario',
        type=int,
        choices=[1, 2, 3, 4],
        help='Run specific scenario (1-4)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all scenarios'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List available scenarios'
    )

    parser.add_argument(
        '--time-limit',
        type=int,
        default=50,
        help='Time limit for solver in seconds (default: 50)'
    )

    args = parser.parse_args()

    try:
        if args.list:
            print_scenarios()
        elif args.all:
            run_all_scenarios(time_limit=args.time_limit)
        elif args.scenario:
            run_scenario(args.scenario, time_limit=args.time_limit)
        else:
            # Default: show menu
            print_scenarios()
            choice = input("Select scenario (1-4) or 'all': ").strip()

            if choice.lower() == 'all':
                run_all_scenarios(time_limit=args.time_limit)
            elif choice in ['1', '2', '3', '4']:
                run_scenario(choice, time_limit=args.time_limit)
            else:
                print(f"Invalid choice: {choice}")
                sys.exit(1)

        print("\n✓ Done!")

    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
