#!/usr/bin/env python3
"""
Data Loader Module for Aircraft Landing Scheduling
Loads problem instances from OR-Library format files
Based on Beasley et al. (2000)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path


@dataclass
class Aircraft:
    """
    Represents a single aircraft with its landing constraints and costs.

    Attributes:
        id: Aircraft identifier (1-indexed to match paper)
        appearance_time: Earliest time aircraft can land (E_i)
        target_time: Preferred landing time (T_i)
        latest_time: Latest time aircraft can land (L_i)
        early_penalty: Cost per time unit for landing before target (g_i)
        late_penalty: Cost per time unit for landing after target (h_i)
    """
    id: int
    appearance_time: float
    target_time: float
    latest_time: float
    early_penalty: float
    late_penalty: float

    def __post_init__(self):
        """Validate aircraft data consistency."""
        assert self.appearance_time <= self.target_time <= self.latest_time, \
            f"Aircraft {self.id}: Time window violation E <= T <= L"
        assert self.early_penalty >= 0 and self.late_penalty >= 0, \
            f"Aircraft {self.id}: Penalties must be non-negative"

    def get_time_window(self) -> Tuple[float, float]:
        """Return the valid landing time window [E_i, L_i]."""
        return (self.appearance_time, self.latest_time)

    def calculate_cost(self, landing_time: float) -> float:
        """
        Calculate the cost of landing at a specific time.

        Args:
            landing_time: Proposed landing time

        Returns:
            Total cost based on deviation from target time
        """
        if landing_time < self.appearance_time or landing_time > self.latest_time:
            return float('inf')  # Infeasible

        if landing_time < self.target_time:
            # Early landing
            return self.early_penalty * (self.target_time - landing_time)
        elif landing_time > self.target_time:
            # Late landing
            return self.late_penalty * (landing_time - self.target_time)
        else:
            # On-time landing
            return 0.0

    def __repr__(self) -> str:
        return (f"Aircraft(id={self.id}, E={self.appearance_time}, "
                f"T={self.target_time}, L={self.latest_time})")


@dataclass
class ProblemInstance:
    """
    Represents a complete aircraft landing problem instance.

    Attributes:
        aircraft: List of Aircraft objects
        separation_matrix: P×P matrix where S[i][j] is minimum separation
                          time required when aircraft i lands before j
        freeze_time: Time before which schedule cannot be changed (optional)
    """
    aircraft: List[Aircraft]
    separation_matrix: np.ndarray
    freeze_time: float = 0.0

    def __post_init__(self):
        """Validate problem instance consistency."""
        n = len(self.aircraft)
        assert self.separation_matrix.shape == (n, n), \
            "Separation matrix dimensions must match number of aircraft"
        assert np.all(self.separation_matrix >= 0), \
            "Separation times must be non-negative"

    @property
    def num_aircraft(self) -> int:
        """Return the number of aircraft in the problem."""
        return len(self.aircraft)

    def get_separation(self, i: int, j: int) -> float:
        """
        Get required separation time when aircraft i lands before j.

        Args:
            i: Index of first aircraft (0-indexed)
            j: Index of second aircraft (0-indexed)

        Returns:
            Minimum separation time S_ij
        """
        return self.separation_matrix[i, j]

    def get_aircraft_by_id(self, aircraft_id: int) -> Optional[Aircraft]:
        """Get aircraft by ID (1-indexed)."""
        for aircraft in self.aircraft:
            if aircraft.id == aircraft_id:
                return aircraft
        return None


class DataLoader:
    """
    Loads aircraft landing problem instances from OR-Library format files.

    File format (from Beasley et al. 2000):
        Line 1: Number of aircraft P, Freeze time
        Lines 2 to P+1: For each aircraft i:
            appearance_time, target_time, latest_time, early_penalty, late_penalty,
            separation_times_when_i_before_others (P values)
    """

    @staticmethod
    def load_from_file(filepath: str) -> ProblemInstance:
        """
        Load problem instance from OR-Library format file.

        Args:
            filepath: Path to the problem instance file

        Returns:
            ProblemInstance object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Instance file not found: {filepath}")

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Remove comments and empty lines
        lines = [line.strip() for line in lines
                if line.strip() and not line.strip().startswith('#')]

        if len(lines) < 1:
            raise ValueError("Empty or invalid file")

        # Parse header
        header = lines[0].split()
        num_aircraft = int(header[0])
        freeze_time = float(header[1]) if len(header) > 1 else 0.0

        if len(lines) < num_aircraft + 1:
            raise ValueError(f"File has {len(lines)} lines but expected "
                           f"{num_aircraft + 1} (header + {num_aircraft} aircraft)")

        # Parse aircraft data
        aircraft_list = []
        separation_matrix = np.zeros((num_aircraft, num_aircraft))

        for i in range(num_aircraft):
            data = lines[i + 1].split()

            if len(data) < 5 + num_aircraft:
                raise ValueError(f"Aircraft {i+1}: Expected at least "
                               f"{5 + num_aircraft} values, got {len(data)}")

            # Parse aircraft parameters
            aircraft = Aircraft(
                id=i + 1,  # 1-indexed as in the paper
                appearance_time=float(data[0]),
                target_time=float(data[1]),
                latest_time=float(data[2]),
                early_penalty=float(data[3]),
                late_penalty=float(data[4])
            )
            aircraft_list.append(aircraft)

            # Parse separation times (when aircraft i lands before others)
            for j in range(num_aircraft):
                separation_matrix[i, j] = float(data[5 + j])

        instance = ProblemInstance(
            aircraft=aircraft_list,
            separation_matrix=separation_matrix,
            freeze_time=freeze_time
        )

        return instance

    @staticmethod
    def save_to_file(instance: ProblemInstance, filepath: str, instance_name: str = "sample"):
        """
        Save a problem instance to OR-Library format file.

        Args:
            instance: Problem instance to save
            filepath: Path to save the file
            instance_name: Name for the instance (for header comment)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            # Write header comments
            f.write(f"# Aircraft Landing Problem Instance - {instance_name}\n")
            f.write(f"# {instance.num_aircraft} aircraft, 1 runway\n")
            f.write("# Format: num_aircraft freeze_time\n")
            f.write("# For each aircraft: appearance_time target_time latest_time "
                   "early_penalty late_penalty sep_1 sep_2 ... sep_n\n")

            # Write number of aircraft and freeze time
            f.write(f"{instance.num_aircraft} {instance.freeze_time}\n")

            # Write each aircraft data
            for i, aircraft in enumerate(instance.aircraft):
                # Basic data
                f.write(f"{aircraft.appearance_time:.0f} ")
                f.write(f"{aircraft.target_time:.0f} ")
                f.write(f"{aircraft.latest_time:.0f} ")
                f.write(f"{aircraft.early_penalty:.0f} ")
                f.write(f"{aircraft.late_penalty:.0f}")

                # Separation times
                for j in range(instance.num_aircraft):
                    sep = instance.get_separation(i, j)
                    f.write(f" {sep:.0f}")

                f.write("\n")

        print(f"Instance saved to: {filepath}")

    @staticmethod
    def validate_instance(instance: ProblemInstance) -> Tuple[bool, List[str]]:
        """
        Validate problem instance for consistency and feasibility.

        Args:
            instance: Problem instance to validate

        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []

        # Check time windows
        for aircraft in instance.aircraft:
            if aircraft.target_time < aircraft.appearance_time:
                warnings.append(f"Aircraft {aircraft.id}: Target before appearance")
            if aircraft.latest_time < aircraft.target_time:
                warnings.append(f"Aircraft {aircraft.id}: Latest before target")

        # Check separation matrix symmetry (if applicable)
        n = instance.num_aircraft
        for i in range(n):
            for j in range(n):
                if i != j:
                    sep_ij = instance.get_separation(i, j)
                    sep_ji = instance.get_separation(j, i)
                    if sep_ij == 0 and sep_ji == 0:
                        warnings.append(f"Zero separation between aircraft "
                                      f"{i+1} and {j+1}")

        # Check for overlapping time windows with tight separations
        for i in range(n):
            for j in range(i+1, n):
                a1, a2 = instance.aircraft[i], instance.aircraft[j]
                sep = max(instance.get_separation(i, j),
                         instance.get_separation(j, i))

                # Check if time windows allow for separation
                if (a1.latest_time < a2.appearance_time + sep and
                    a2.latest_time < a1.appearance_time + sep):
                    warnings.append(f"Potentially infeasible separation between "
                                  f"aircraft {a1.id} and {a2.id}")

        is_valid = len(warnings) == 0
        return is_valid, warnings

    @staticmethod
    def create_sample_instance(num_aircraft: int = 10, seed: int = 42) -> ProblemInstance:
        """
        Create a random sample problem instance for testing.

        Args:
            num_aircraft: Number of aircraft to generate
            seed: Random seed for reproducibility

        Returns:
            ProblemInstance object
        """
        np.random.seed(seed)

        aircraft_list = []
        for i in range(num_aircraft):
            # Generate time windows
            target = np.random.uniform(0, 100)
            early_buffer = np.random.uniform(10, 30)
            late_buffer = np.random.uniform(10, 30)

            aircraft = Aircraft(
                id=i + 1,
                appearance_time=max(0, target - early_buffer),
                target_time=target,
                latest_time=target + late_buffer,
                early_penalty=np.random.uniform(1, 10),
                late_penalty=np.random.uniform(1, 10)
            )
            aircraft_list.append(aircraft)

        # Generate separation matrix
        # Typically, larger aircraft require more separation
        base_separation = 3.0
        separation_matrix = np.random.uniform(
            base_separation,
            base_separation * 2,
            (num_aircraft, num_aircraft)
        )
        np.fill_diagonal(separation_matrix, 0)  # No separation with self

        instance = ProblemInstance(
            aircraft=aircraft_list,
            separation_matrix=separation_matrix
        )

        return instance

    @staticmethod
    def print_instance_summary(instance: ProblemInstance):
        """Print a human-readable summary of the problem instance."""
        print(f"Problem Instance Summary")
        print(f"=" * 60)
        print(f"Number of aircraft: {instance.num_aircraft}")
        print(f"Freeze time: {instance.freeze_time}")
        print(f"\nAircraft Details:")
        print(f"{'ID':<4} {'E':>8} {'T':>8} {'L':>8} {'g':>8} {'h':>8}")
        print(f"-" * 50)

        for aircraft in instance.aircraft:
            print(f"{aircraft.id:<4} "
                  f"{aircraft.appearance_time:>8.1f} "
                  f"{aircraft.target_time:>8.1f} "
                  f"{aircraft.latest_time:>8.1f} "
                  f"{aircraft.early_penalty:>8.2f} "
                  f"{aircraft.late_penalty:>8.2f}")

        print(f"\nSeparation Matrix (first 5x5):")
        n = min(5, instance.num_aircraft)
        print("    ", end="")
        for j in range(n):
            print(f"{j+1:>6}", end="")
        print()

        for i in range(n):
            print(f"{i+1:>3} ", end="")
            for j in range(n):
                print(f"{instance.separation_matrix[i,j]:>6.1f}", end="")
            print()

        if instance.num_aircraft > 5:
            print(f"... (showing first 5 of {instance.num_aircraft} aircraft)")


if __name__ == "__main__":
    # Test the data loader
    print("Testing DataLoader module...")

    # Create sample instance
    print("\n1. Creating sample instance...")
    instance = DataLoader.create_sample_instance(num_aircraft=5)
    DataLoader.print_instance_summary(instance)

    # Validate instance
    print("\n2. Validating instance...")
    is_valid, warnings = DataLoader.validate_instance(instance)
    print(f"Valid: {is_valid}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    print("\nDataLoader module test completed!")
