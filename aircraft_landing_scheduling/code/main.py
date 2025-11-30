#!/usr/bin/env python3
"""
Main Entry Point for Aircraft Landing Scheduling
Command-line interface for running experiments and generating results
"""

import argparse
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.data_loader import DataLoader, ProblemInstance
from code.solver import OptimalSolver, BatchSolver
from code.visualization import ResultVisualizer
from code.utils import (
    print_solution_details,
    export_solution_json,
    export_detailed_solution_table,
    validate_solution,
    print_table
)


def run_single_instance(args):
    """
    Run single instance experiment.

    Args:
        args: Parsed command-line arguments
    """
    print("="*70)
    print("AIRCRAFT LANDING SCHEDULING - SINGLE INSTANCE")
    print("="*70)

    # Load instance
    if args.instance == "sample":
        print(f"\nCreating sample instance with {args.num_aircraft} aircraft...")
        instance = DataLoader.create_sample_instance(
            num_aircraft=args.num_aircraft,
            seed=args.seed
        )
        instance_name = f"sample_{args.num_aircraft}"

        # Save sample instance to data folder
        data_file = Path("data") / f"sample_{args.num_aircraft}_seed{args.seed}.txt"
        DataLoader.save_to_file(instance, str(data_file), instance_name)
    else:
        print(f"\nLoading instance: {args.instance}")
        instance = DataLoader.load_from_file(args.instance)
        instance_name = Path(args.instance).stem

    # Print instance info
    if args.verbose:
        DataLoader.print_instance_summary(instance)

    # Create solver
    solver = OptimalSolver(instance, instance_name=instance_name)

    # Solve and compare
    result = solver.solve_and_compare(
        num_runways=args.runways,
        time_limit=args.time_limit,
        use_multi_start=args.multi_start
    )

    # Visualize results
    if not args.no_viz:
        print("\nGenerating visualizations...")
        viz = ResultVisualizer(output_dir=args.output_dir)

        # Gantt charts
        viz.plot_gantt_chart(
            instance,
            result.heuristic_solution,
            title=f"{instance_name} - Heuristic Solution",
            filename=f"{instance_name}_heuristic_gantt"
        )

        viz.plot_gantt_chart(
            instance,
            result.optimal_solution,
            title=f"{instance_name} - Optimal Solution",
            filename=f"{instance_name}_optimal_gantt"
        )

        # Comparison plot
        viz.plot_comparison(
            instance,
            result.heuristic_solution,
            result.optimal_solution,
            filename=f"{instance_name}_comparison"
        )

        # Cost breakdown
        viz.plot_cost_breakdown(
            instance,
            result.optimal_solution,
            filename=f"{instance_name}_cost_breakdown"
        )

    # Validate solution
    if args.validate:
        print("\nValidating optimal solution...")
        is_valid, violations = validate_solution(instance, result.optimal_solution)
        if is_valid:
            print("✓ Solution is valid!")
        else:
            print("✗ Solution has violations:")
            for v in violations:
                print(f"  - {v}")

    # Export results
    if args.export_json:
        json_path = Path(args.output_dir) / "tables" / f"{instance_name}_optimal.json"
        export_solution_json(instance, result.optimal_solution, str(json_path))

    # Always export detailed CSV tables
    print("\nExporting detailed solution tables...")
    heuristic_csv = Path(args.output_dir) / "tables" / f"{instance_name}_heuristic_detailed.csv"
    optimal_csv = Path(args.output_dir) / "tables" / f"{instance_name}_optimal_detailed.csv"

    export_detailed_solution_table(instance, result.heuristic_solution, str(heuristic_csv), "Heuristic")
    export_detailed_solution_table(instance, result.optimal_solution, str(optimal_csv), "Optimal")

    print("\n✓ Single instance experiment completed!")


def run_batch_experiment(args):
    """
    Run batch experiment on multiple instances.

    Args:
        args: Parsed command-line arguments
    """
    print("="*70)
    print("AIRCRAFT LANDING SCHEDULING - BATCH EXPERIMENT")
    print("="*70)

    if not args.data:
        print("Error: --data directory required for batch mode")
        sys.exit(1)

    # Create batch solver
    batch_solver = BatchSolver(args.data)

    # Run experiments
    output_file = Path(args.output_dir) / "tables" / f"batch_results_r{args.runways}.csv"

    df = batch_solver.run_batch_experiment(
        num_runways=args.runways,
        time_limit=args.time_limit,
        use_multi_start=args.multi_start,
        output_file=str(output_file)
    )

    if not df.empty and not args.no_viz:
        print("\nGenerating summary visualizations...")

        # Create comparison plots for batch results
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Gap comparison
        ax = axes[0, 0]
        df_sorted = df.sort_values('Gap (%)')
        ax.barh(df_sorted['Instance'], df_sorted['Gap (%)'],
               color='skyblue', edgecolor='black')
        ax.set_xlabel('Gap (%)', fontweight='bold')
        ax.set_title('Heuristic Gap by Instance', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # 2. Speedup comparison
        ax = axes[0, 1]
        df_sorted = df.sort_values('Speedup')
        ax.barh(df_sorted['Instance'], df_sorted['Speedup'],
               color='lightgreen', edgecolor='black')
        ax.set_xlabel('Speedup (x)', fontweight='bold')
        ax.set_title('Computation Speedup', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # 3. Cost comparison
        ax = axes[1, 0]
        x = range(len(df))
        width = 0.35
        ax.bar([i - width/2 for i in x], df['Heuristic Cost'], width,
              label='Heuristic', color='skyblue', edgecolor='black')
        ax.bar([i + width/2 for i in x], df['Optimal Cost'], width,
              label='Optimal', color='lightgreen', edgecolor='black')
        ax.set_xlabel('Instance', fontweight='bold')
        ax.set_ylabel('Cost', fontweight='bold')
        ax.set_title('Cost Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Instance'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 4. Summary statistics
        ax = axes[1, 1]
        ax.axis('off')

        summary_data = [
            ['Instances', len(df)],
            ['Avg Gap (%)', f"{df['Gap (%)'].mean():.2f}"],
            ['Max Gap (%)', f"{df['Gap (%)'].max():.2f}"],
            ['Avg Speedup', f"{df['Speedup'].mean():.1f}x"],
            ['Total Opt Time', f"{df['Optimal Time (s)'].sum():.1f}s"],
            ['Total Heur Time', f"{df['Heuristic Time (s)'].sum():.3f}s"]
        ]

        from tabulate import tabulate
        table_str = tabulate(summary_data, tablefmt='simple')
        ax.text(0.1, 0.5, table_str, transform=ax.transAxes,
               fontsize=11, verticalalignment='center',
               fontfamily='monospace')
        ax.set_title('Summary Statistics', fontweight='bold', pad=20)

        plt.suptitle(f'Batch Experiment Results ({args.runways} runway(s))',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        viz_path = Path(args.output_dir) / "figures" / f"batch_summary_r{args.runways}.png"
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"Summary visualization saved: {viz_path}")
        plt.close()

    print("\n✓ Batch experiment completed!")


def run_runway_analysis(args):
    """
    Analyze effect of multiple runways.

    Args:
        args: Parsed command-line arguments
    """
    print("="*70)
    print("AIRCRAFT LANDING SCHEDULING - RUNWAY ANALYSIS")
    print("="*70)

    # Load instance
    if args.instance == "sample":
        print(f"\nCreating sample instance with {args.num_aircraft} aircraft...")
        instance = DataLoader.create_sample_instance(
            num_aircraft=args.num_aircraft,
            seed=args.seed
        )
        instance_name = f"sample_{args.num_aircraft}"

        # Save sample instance to data folder
        data_file = Path("data") / f"sample_{args.num_aircraft}_seed{args.seed}.txt"
        DataLoader.save_to_file(instance, str(data_file), instance_name)
    else:
        print(f"\nLoading instance: {args.instance}")
        instance = DataLoader.load_from_file(args.instance)
        instance_name = Path(args.instance).stem

    # Create solver
    solver = OptimalSolver(instance, instance_name=instance_name)

    # Run runway analysis
    output_file = Path(args.output_dir) / "tables" / f"{instance_name}_runway_analysis.csv"

    df = solver.solve_multiple_runways(
        max_runways=args.max_runways,
        time_limit=args.time_limit
    )

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Visualize
    if not args.no_viz:
        print("\nGenerating runway analysis visualization...")
        viz = ResultVisualizer(output_dir=args.output_dir)
        viz.plot_runway_analysis(
            df,
            filename=f"{instance_name}_runway_analysis"
        )

    print("\n✓ Runway analysis completed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aircraft Landing Scheduling Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run sample instance with 1 runway
  python main.py --instance sample --runways 1

  # Run specific instance file
  python main.py --instance data/airland1.txt --runways 1

  # Run batch experiment on all instances
  python main.py --batch --data data/ --runways 1

  # Analyze effect of multiple runways
  python main.py --runway-analysis --instance sample --max-runways 4

  # Run with custom parameters
  python main.py --instance sample --runways 2 --time-limit 600 --multi-start
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--batch',
        action='store_true',
        help='Run batch experiment on multiple instances'
    )
    mode_group.add_argument(
        '--runway-analysis',
        action='store_true',
        help='Analyze effect of multiple runways'
    )

    # Input options
    parser.add_argument(
        '--instance',
        type=str,
        default='sample',
        help='Instance file path or "sample" for random instance (default: sample)'
    )
    parser.add_argument(
        '--data',
        type=str,
        help='Data directory for batch mode'
    )
    parser.add_argument(
        '--num-aircraft',
        type=int,
        default=10,
        help='Number of aircraft for sample instance (default: 10)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sample instance (default: 42)'
    )

    # Solver options
    parser.add_argument(
        '--runways',
        type=int,
        default=1,
        help='Number of runways (default: 1)'
    )
    parser.add_argument(
        '--max-runways',
        type=int,
        default=4,
        help='Maximum runways for runway analysis (default: 4)'
    )
    parser.add_argument(
        '--time-limit',
        type=int,
        default=300,
        help='Time limit in seconds for MIP solver (default: 300)'
    )
    parser.add_argument(
        '--multi-start',
        action='store_true',
        help='Use multi-start heuristic'
    )

    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualization'
    )
    parser.add_argument(
        '--export-json',
        action='store_true',
        help='Export solution to JSON'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate solution feasibility'
    )

    # Display options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Execute appropriate mode
    try:
        start_time = time.time()

        if args.batch:
            run_batch_experiment(args)
        elif args.runway_analysis:
            run_runway_analysis(args)
        else:
            run_single_instance(args)

        total_time = time.time() - start_time
        print(f"\nTotal execution time: {total_time:.2f}s")

    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
