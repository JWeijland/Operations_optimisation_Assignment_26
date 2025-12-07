#!/usr/bin/env python3
"""
Configuration file for Sensitivity Analysis
Allows easy customization of scenario parameters and experiment settings
"""

from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Warning: PyYAML not installed. Install with: pip install pyyaml")
    yaml = None


@dataclass
class RushHourConfig:
    """Configuration for rush hour traffic pattern."""
    start_minute: float = 60.0  # Start of rush hour (minutes from base time)
    end_minute: float = 90.0    # End of rush hour
    probability: float = 0.5    # Probability aircraft arrives in rush hour (0.0-1.0)

    def __post_init__(self):
        assert 0.0 <= self.probability <= 1.0, "probability must be between 0 and 1"
        assert self.start_minute < self.end_minute, "start must be before end"


@dataclass
class AircraftMixConfig:
    """Configuration for aircraft type distribution."""
    heavy_ratio: float = 0.30
    medium_ratio: float = 0.60
    light_ratio: float = 0.10

    def __post_init__(self):
        total = self.heavy_ratio + self.medium_ratio + self.light_ratio
        assert abs(total - 1.0) < 0.01, f"Ratios must sum to 1.0, got {total}"


@dataclass
class TimeWindowConfig:
    """Configuration for time window (2 hours)."""
    total_minutes: float = 120.0  # Total time window (2 hours)
    base_time_label: str = "18:00"  # Label for base time (e.g., "18:00")


@dataclass
class SensitivityAnalysisConfig:
    """Complete configuration for sensitivity analysis experiment."""

    # Experiment name
    experiment_name: str = "sensitivity_analysis"

    # Aircraft counts to test
    aircraft_counts: List[int] = None

    # Runway counts to test
    runway_counts: List[int] = None

    # Rush hour configuration
    rush_hour: RushHourConfig = None

    # Aircraft mix configuration
    aircraft_mix: AircraftMixConfig = None

    # Time window configuration
    time_window: TimeWindowConfig = None

    # Solver settings
    time_limit: int = 50  # seconds per instance
    random_seed: int = 42

    # Number of replications per scenario
    num_replications: int = 1

    # Output settings
    output_dir: str = "results/sensitivity_analysis"
    save_detailed_results: bool = True
    generate_heatmaps: bool = True

    def __post_init__(self):
        # Set defaults if not provided
        if self.aircraft_counts is None:
            self.aircraft_counts = [20, 30, 40, 50]

        if self.runway_counts is None:
            self.runway_counts = [1, 2, 3, 4]

        if self.rush_hour is None:
            self.rush_hour = RushHourConfig()

        if self.aircraft_mix is None:
            self.aircraft_mix = AircraftMixConfig()

        if self.time_window is None:
            self.time_window = TimeWindowConfig()

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'experiment_name': self.experiment_name,
            'aircraft_counts': self.aircraft_counts,
            'runway_counts': self.runway_counts,
            'rush_hour': {
                'start_minute': self.rush_hour.start_minute,
                'end_minute': self.rush_hour.end_minute,
                'probability': self.rush_hour.probability
            },
            'aircraft_mix': {
                'heavy_ratio': self.aircraft_mix.heavy_ratio,
                'medium_ratio': self.aircraft_mix.medium_ratio,
                'light_ratio': self.aircraft_mix.light_ratio
            },
            'time_window': {
                'total_minutes': self.time_window.total_minutes,
                'base_time_label': self.time_window.base_time_label
            },
            'solver': {
                'time_limit': self.time_limit,
                'random_seed': self.random_seed
            },
            'replications': self.num_replications,
            'output': {
                'output_dir': self.output_dir,
                'save_detailed_results': self.save_detailed_results,
                'generate_heatmaps': self.generate_heatmaps
            }
        }

    def save_to_yaml(self, filepath: str):
        """Save configuration to YAML file."""
        if yaml is None:
            print("ERROR: PyYAML not installed. Cannot save to YAML.")
            return

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

        print(f"Configuration saved to: {filepath}")

    @staticmethod
    def load_from_yaml(filepath: str) -> 'SensitivityAnalysisConfig':
        """Load configuration from YAML file."""
        if yaml is None:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")

        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        # Reconstruct config object
        config = SensitivityAnalysisConfig(
            experiment_name=data['experiment_name'],
            aircraft_counts=data['aircraft_counts'],
            runway_counts=data['runway_counts'],
            rush_hour=RushHourConfig(**data['rush_hour']),
            aircraft_mix=AircraftMixConfig(**data['aircraft_mix']),
            time_window=TimeWindowConfig(**data['time_window']),
            time_limit=data['solver']['time_limit'],
            random_seed=data['solver']['random_seed'],
            num_replications=data['replications'],
            output_dir=data['output']['output_dir'],
            save_detailed_results=data['output']['save_detailed_results'],
            generate_heatmaps=data['output']['generate_heatmaps']
        )

        return config

    def print_summary(self):
        """Print a summary of the configuration."""
        print("=" * 70)
        print("SENSITIVITY ANALYSIS CONFIGURATION")
        print("=" * 70)
        print(f"Experiment: {self.experiment_name}")
        print(f"\nScenarios:")
        print(f"  Aircraft counts: {self.aircraft_counts}")
        print(f"  Runway counts: {self.runway_counts}")
        print(f"  Total scenarios: {len(self.aircraft_counts)} × {len(self.runway_counts)} = "
              f"{len(self.aircraft_counts) * len(self.runway_counts)}")
        print(f"  Replications per scenario: {self.num_replications}")
        print(f"\nRush Hour Configuration:")
        print(f"  Time window: {self.rush_hour.start_minute:.0f}-{self.rush_hour.end_minute:.0f} minutes")
        print(f"  Rush hour probability: {self.rush_hour.probability*100:.0f}%")
        print(f"\nAircraft Mix:")
        print(f"  Heavy: {self.aircraft_mix.heavy_ratio*100:.0f}%")
        print(f"  Medium: {self.aircraft_mix.medium_ratio*100:.0f}%")
        print(f"  Light: {self.aircraft_mix.light_ratio*100:.0f}%")
        print(f"\nTime Window:")
        print(f"  Total duration: {self.time_window.total_minutes:.0f} minutes")
        print(f"  Base time: {self.time_window.base_time_label}")
        print(f"\nSolver Settings:")
        print(f"  Time limit: {self.time_limit}s per instance")
        print(f"  Random seed: {self.random_seed}")
        print(f"\nOutput:")
        print(f"  Directory: {self.output_dir}")
        print(f"  Save detailed results: {self.save_detailed_results}")
        print(f"  Generate heatmaps: {self.generate_heatmaps}")
        print("=" * 70)


# Predefined configurations for common experiments

def create_default_config() -> SensitivityAnalysisConfig:
    """Create default configuration."""
    return SensitivityAnalysisConfig(
        experiment_name="default_sensitivity",
        aircraft_counts=[20, 30, 40, 50],
        runway_counts=[1, 2, 3, 4],
        rush_hour=RushHourConfig(
            start_minute=60.0,
            end_minute=90.0,
            probability=0.5
        ),
        aircraft_mix=AircraftMixConfig(
            heavy_ratio=0.30,
            medium_ratio=0.60,
            light_ratio=0.10
        )
    )


def create_extreme_rush_config() -> SensitivityAnalysisConfig:
    """Create configuration with extreme rush hour (90% concentration).
    NOTE: Aircraft mix ratio's blijven vast zoals default (30/60/10)."""
    return SensitivityAnalysisConfig(
        experiment_name="extreme_rush_sensitivity",
        aircraft_counts=[20, 30, 40, 50],
        runway_counts=[1, 2, 3, 4],
        rush_hour=RushHourConfig(
            start_minute=60.0,
            end_minute=90.0,
            probability=0.9  # 90% in rush hour!
        ),
        aircraft_mix=AircraftMixConfig(
            heavy_ratio=0.30,  # FIXED: Same as default
            medium_ratio=0.60,  # FIXED: Same as default
            light_ratio=0.10   # FIXED: Same as default
        )
    )


def create_no_rush_config() -> SensitivityAnalysisConfig:
    """Create configuration with uniform distribution (no rush hour)."""
    return SensitivityAnalysisConfig(
        experiment_name="no_rush_sensitivity",
        aircraft_counts=[20, 30, 40, 50],
        runway_counts=[1, 2, 3, 4],
        rush_hour=RushHourConfig(
            start_minute=60.0,
            end_minute=90.0,
            probability=0.25  # Uniform distribution
        )
    )


def create_heavy_traffic_config() -> SensitivityAnalysisConfig:
    """Create configuration focused on heavy aircraft.
    NOTE: Aircraft mix ratio's blijven vast zoals default (30/60/10)."""
    return SensitivityAnalysisConfig(
        experiment_name="heavy_traffic_sensitivity",
        aircraft_counts=[20, 30, 40, 50],
        runway_counts=[1, 2, 3, 4],
        rush_hour=RushHourConfig(
            start_minute=60.0,
            end_minute=90.0,
            probability=0.6
        ),
        aircraft_mix=AircraftMixConfig(
            heavy_ratio=0.30,  # FIXED: Same as default
            medium_ratio=0.60,  # FIXED: Same as default
            light_ratio=0.10   # FIXED: Same as default
        )
    )


if __name__ == "__main__":
    # Example: Create and save default configuration
    config = create_default_config()
    config.print_summary()
    config.save_to_yaml("configs/default_sensitivity.yaml")

    # Example: Create extreme rush hour configuration
    extreme_config = create_extreme_rush_config()
    extreme_config.save_to_yaml("configs/extreme_rush_sensitivity.yaml")

    print("\n✓ Example configurations created in configs/ directory")
