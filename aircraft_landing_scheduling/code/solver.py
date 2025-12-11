#!/usr/bin/env python3
"""
Solver Module - Combines optimal and heuristic approaches
Provides high-level interface for solving aircraft landing problems
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import time

try:
    from .data_loader import ProblemInstance, DataLoader
    from .model import AircraftLandingModel, Solution
    from .heuristic import GreedyHeuristic, MultiStartGreedy
except ImportError:
    from data_loader import ProblemInstance, DataLoader
    from model import AircraftLandingModel, Solution
    from heuristic import GreedyHeuristic, MultiStartGreedy


@dataclass
class ComparisonResult:
    """
    Stores comparison between heuristic and optimal solutions.

    Attributes:
        instance_name: Name of problem instance
        num_aircraft: Number of aircraft
        num_runways: Number of runways
        heuristic_cost: Cost of heuristic solution
        optimal_cost: Cost of optimal solution
        heuristic_time: Time to solve heuristic (seconds)
        optimal_time: Time to solve optimal (seconds)
        gap: Quality gap = (heuristic - optimal) / optimal * 100%
        speedup: Time speedup = optimal_time / heuristic_time
    """
    instance_name: str
    num_aircraft: int
    num_runways: int
    heuristic_cost: float
    optimal_cost: float
    heuristic_time: float
    optimal_time: float
    gap: float
    speedup: float
    heuristic_solution: Optional[Solution] = None
    optimal_solution: Optional[Solution] = None

    def __repr__(self) -> str:
        return (f"ComparisonResult({self.instance_name}, "
                f"gap={self.gap:.2f}%, speedup={self.speedup:.1f}x)")

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame export."""
        return {
            'Instance': self.instance_name,
            'Aircraft': self.num_aircraft,
            'Runways': self.num_runways,
            'Heuristic Cost': self.heuristic_cost,
            'Optimal Cost': self.optimal_cost,
            'Gap (%)': self.gap,
            'Heuristic Time (s)': self.heuristic_time,
            'Optimal Time (s)': self.optimal_time,
            'Speedup': self.speedup
        }


class OptimalSolver:
    """
    High-level solver that provides both optimal and heuristic solutions.
    """

    def __init__(self, instance: ProblemInstance, instance_name: str = "unknown"):
        """
        Initialize solver.

        Args:
            instance: Problem instance to solve
            instance_name: Name for reporting
        """
        self.instance = instance
        self.instance_name = instance_name

    def solve_heuristic(
        self,
        num_runways: int = 1,
        multi_start: bool = False,
        num_starts: int = 5
    ) -> Solution:
        """
        Solve using greedy heuristic.

        Args:
            num_runways: Number of runways
            multi_start: Use multi-start strategy
            num_starts: Number of starts if multi_start=True

        Returns:
            Solution object
        """
        if multi_start:
            solver = MultiStartGreedy(self.instance, num_runways)
            solution = solver.solve(num_starts=num_starts)
        else:
            solver = GreedyHeuristic(self.instance, num_runways)
            solution = solver.solve(improve=True)

        return solution

    def solve_optimal(
        self,
        num_runways: int = 1,
        time_limit: int = 300,
        gap: float = 0.0  # 0% gap = optimal solutions only!
    ) -> Optional[Solution]:
        """
        Solve optimally using MIP.

        Args:
            num_runways: Number of runways
            time_limit: Time limit in seconds
            gap: Acceptable optimality gap

        Returns:
            Solution object if solved, None otherwise
        """
        model = AircraftLandingModel(self.instance, num_runways)
        solution = model.solve(
            time_limit=time_limit,
            gap=gap,
            verbose=False
        )

        return solution

    def solve_and_compare(
        self,
        num_runways: int = 1,
        time_limit: int = 300,
        gap: float = 0.0,  # 0% gap = optimal solutions only!
        use_multi_start: bool = False
    ) -> ComparisonResult:
        """
        Solve with both methods and compare results.

        Args:
            num_runways: Number of runways
            time_limit: Time limit for optimal solver
            gap: Optimality gap (default 0.0 = optimal solutions only!)
            use_multi_start: Use multi-start heuristic

        Returns:
            ComparisonResult object
        """
        print(f"\n{'='*70}")
        print(f"Solving: {self.instance_name}")
        print(f"Aircraft: {self.instance.num_aircraft}, Runways: {num_runways}")
        print(f"{'='*70}")

        # Solve with heuristic
        print("\n1. HEURISTIC SOLUTION")
        print("-" * 70)
        heuristic_solution = self.solve_heuristic(
            num_runways=num_runways,
            multi_start=use_multi_start
        )

        # Solve optimally
        print("\n2. OPTIMAL SOLUTION")
        print("-" * 70)
        optimal_solution = self.solve_optimal(
            num_runways=num_runways,
            time_limit=time_limit,
            gap=gap
        )

        if optimal_solution is None:
            print("Warning: Optimal solution not found, using heuristic only")
            optimal_solution = heuristic_solution

        # Compare solutions
        gap = ((heuristic_solution.objective_value - optimal_solution.objective_value) /
               optimal_solution.objective_value * 100) if optimal_solution.objective_value > 0 else 0.0
        speedup = optimal_solution.solve_time / max(heuristic_solution.solve_time, 0.0001)

        result = ComparisonResult(
            instance_name=self.instance_name,
            num_aircraft=self.instance.num_aircraft,
            num_runways=num_runways,
            heuristic_cost=heuristic_solution.objective_value,
            optimal_cost=optimal_solution.objective_value,
            heuristic_time=heuristic_solution.solve_time,
            optimal_time=optimal_solution.solve_time,
            gap=gap,
            speedup=speedup,
            heuristic_solution=heuristic_solution,
            optimal_solution=optimal_solution
        )

        # Print comparison
        print("\n3. COMPARISON")
        print("-" * 70)
        print(f"Heuristic cost:     {heuristic_solution.objective_value:>10.2f}")
        print(f"Optimal cost:       {optimal_solution.objective_value:>10.2f}")
        print(f"Gap:                {gap:>9.2f}%")
        print(f"Heuristic time:     {heuristic_solution.solve_time:>10.3f}s")
        print(f"Optimal time:       {optimal_solution.solve_time:>10.3f}s")
        print(f"Speedup:            {speedup:>10.1f}x")
        print("=" * 70)

        return result

    def solve_multiple_runways(
        self,
        max_runways: int = 4,
        time_limit: int = 300
    ) -> pd.DataFrame:
        """
        Analyze effect of increasing number of runways.

        Args:
            max_runways: Maximum number of runways to test
            time_limit: Time limit per instance

        Returns:
            DataFrame with results
        """
        print(f"\n{'='*70}")
        print(f"RUNWAY ANALYSIS: {self.instance_name}")
        print(f"Testing 1 to {max_runways} runways")
        print(f"{'='*70}")

        results = []

        for num_runways in range(1, max_runways + 1):
            print(f"\n--- Testing {num_runways} runway(s) ---")

            result = self.solve_and_compare(
                num_runways=num_runways,
                time_limit=time_limit
            )

            results.append(result.to_dict())

        df = pd.DataFrame(results)
        return df


class BatchSolver:
    """
    Solver for running experiments on multiple problem instances.
    """

    def __init__(self, data_directory: str):
        """
        Initialize batch solver.

        Args:
            data_directory: Directory containing problem instance files
        """
        self.data_directory = Path(data_directory)
        if not self.data_directory.exists():
            raise FileNotFoundError(f"Data directory not found: {data_directory}")

    def load_instances(self, pattern: str = "*.txt") -> Dict[str, ProblemInstance]:
        """
        Load all problem instances from directory.

        Args:
            pattern: File pattern to match (default: *.txt)

        Returns:
            Dictionary mapping instance name to ProblemInstance
        """
        instances = {}
        files = list(self.data_directory.glob(pattern))

        print(f"Loading instances from {self.data_directory}")
        print(f"Found {len(files)} files")

        for filepath in files:
            try:
                instance = DataLoader.load_from_file(str(filepath))
                instance_name = filepath.stem
                instances[instance_name] = instance
                print(f"  ✓ Loaded: {instance_name} ({instance.num_aircraft} aircraft)")
            except Exception as e:
                print(f"  ✗ Failed to load {filepath.name}: {e}")

        return instances

    def run_batch_experiment(
        self,
        num_runways: int = 1,
        time_limit: int = 300,
        use_multi_start: bool = False,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Run experiments on all instances.

        Args:
            num_runways: Number of runways
            time_limit: Time limit per instance
            use_multi_start: Use multi-start heuristic
            output_file: Save results to CSV file

        Returns:
            DataFrame with all results
        """
        instances = self.load_instances()

        if not instances:
            print("No instances loaded!")
            return pd.DataFrame()

        print(f"\n{'='*70}")
        print(f"BATCH EXPERIMENT")
        print(f"Instances: {len(instances)}")
        print(f"Runways: {num_runways}")
        print(f"Time limit: {time_limit}s per instance")
        print(f"{'='*70}")

        results = []

        for i, (name, instance) in enumerate(instances.items(), 1):
            print(f"\n[{i}/{len(instances)}] Processing: {name}")

            solver = OptimalSolver(instance, instance_name=name)

            try:
                result = solver.solve_and_compare(
                    num_runways=num_runways,
                    time_limit=time_limit,
                    use_multi_start=use_multi_start
                )
                results.append(result.to_dict())

            except Exception as e:
                print(f"Error solving {name}: {e}")
                continue

        # Create DataFrame
        df = pd.DataFrame(results)

        # Calculate summary statistics
        print(f"\n{'='*70}")
        print("SUMMARY STATISTICS")
        print(f"{'='*70}")
        if not df.empty:
            print(f"Total instances:    {len(df)}")
            print(f"Average gap:        {df['Gap (%)'].mean():.2f}%")
            print(f"Max gap:            {df['Gap (%)'].max():.2f}%")
            print(f"Average speedup:    {df['Speedup'].mean():.1f}x")
            print(f"Total opt time:     {df['Optimal Time (s)'].sum():.1f}s")
            print(f"Total heur time:    {df['Heuristic Time (s)'].sum():.1f}s")

        # Save to file
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")

        return df

    def run_runway_analysis(
        self,
        instance_name: str,
        max_runways: int = 4,
        time_limit: int = 300,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Analyze effect of runways on a specific instance.

        Args:
            instance_name: Name of instance to analyze
            max_runways: Maximum number of runways
            time_limit: Time limit per solve
            output_file: Save results to CSV

        Returns:
            DataFrame with runway analysis results
        """
        instances = self.load_instances()

        if instance_name not in instances:
            raise ValueError(f"Instance not found: {instance_name}")

        instance = instances[instance_name]
        solver = OptimalSolver(instance, instance_name=instance_name)

        df = solver.solve_multiple_runways(
            max_runways=max_runways,
            time_limit=time_limit
        )

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"\nRunway analysis saved to: {output_path}")

        return df
