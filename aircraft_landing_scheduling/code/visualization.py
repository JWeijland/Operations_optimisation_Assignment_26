#!/usr/bin/env python3
"""
Visualization Module for Aircraft Landing Scheduling
Creates plots and charts for solution analysis
"""

from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

from .data_loader import ProblemInstance, Aircraft
from .model import Solution


class ResultVisualizer:
    """
    Creates visualizations for aircraft landing scheduling solutions.

    Visualizations include:
    - Gantt charts showing landing schedules
    - Cost breakdown charts
    - Time deviation plots
    - Comparison tables
    """

    def __init__(self, output_dir: str = "results/figures"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10

    def plot_gantt_chart(
        self,
        instance: ProblemInstance,
        solution: Solution,
        title: str = "Aircraft Landing Schedule",
        filename: Optional[str] = None
    ):
        """
        Create Gantt chart showing landing schedule.

        Args:
            instance: Problem instance
            solution: Solution to visualize
            title: Chart title
            filename: Output filename (without extension)
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        # Get schedule organized by runway
        runway_schedule = solution.get_schedule_by_runway()
        num_runways = len(runway_schedule)

        # Color map for aircraft
        colors = plt.cm.tab20(np.linspace(0, 1, len(instance.aircraft)))

        # Plot each runway
        for runway_idx, (runway, landings) in enumerate(sorted(runway_schedule.items())):
            y_position = num_runways - runway_idx  # Invert for top-to-bottom

            for aircraft_id, landing_time in landings:
                # Get aircraft data
                aircraft = instance.get_aircraft_by_id(aircraft_id)

                # Plot time window (light gray bar)
                window_width = aircraft.latest_time - aircraft.appearance_time
                ax.barh(y_position, window_width,
                       left=aircraft.appearance_time,
                       height=0.3,
                       color='lightgray',
                       alpha=0.3,
                       edgecolor='gray',
                       linewidth=0.5)

                # Plot target time (vertical line)
                ax.plot([aircraft.target_time, aircraft.target_time],
                       [y_position - 0.2, y_position + 0.2],
                       'g--', linewidth=2, alpha=0.7)

                # Plot actual landing time (circle)
                color = colors[aircraft_id - 1]
                ax.plot(landing_time, y_position,
                       'o', markersize=12, color=color,
                       markeredgecolor='black', markeredgewidth=1.5)

                # Add aircraft label
                ax.text(landing_time, y_position + 0.35,
                       f'A{aircraft_id}',
                       ha='center', va='bottom',
                       fontsize=8, fontweight='bold')

                # Add time label
                deviation = landing_time - aircraft.target_time
                time_str = f'{landing_time:.1f}'
                if abs(deviation) > 0.1:
                    time_str += f'\n({deviation:+.1f})'

                ax.text(landing_time, y_position - 0.35,
                       time_str,
                       ha='center', va='top',
                       fontsize=7)

        # Configure axes
        ax.set_ylim(0.5, num_runways + 0.5)
        ax.set_yticks(range(1, num_runways + 1))
        ax.set_yticklabels([f'Runway {i}' for i in range(1, num_runways + 1)])

        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Runway', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor='lightgray', edgecolor='gray',
                          alpha=0.3, label='Time Window [E, L]'),
            plt.Line2D([0], [0], color='g', linestyle='--',
                      linewidth=2, label='Target Time (T)'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='blue', markersize=10,
                      markeredgecolor='black', markeredgewidth=1.5,
                      linestyle='', label='Actual Landing')
        ]
        ax.legend(handles=legend_elements, loc='upper right',
                 fontsize=10, framealpha=0.9)

        # Add cost annotation
        cost_text = f'Total Cost: {solution.objective_value:.2f}\n'
        cost_text += f'Status: {solution.status}\n'
        cost_text += f'Solve Time: {solution.solve_time:.3f}s'

        ax.text(0.02, 0.98, cost_text,
               transform=ax.transAxes,
               fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        # Save or show
        if filename:
            filepath = self.output_dir / f"{filename}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Gantt chart saved: {filepath}")
        else:
            plt.show()

        plt.close()

    def plot_cost_breakdown(
        self,
        instance: ProblemInstance,
        solution: Solution,
        filename: Optional[str] = None
    ):
        """
        Create bar chart showing cost breakdown by aircraft.

        Args:
            instance: Problem instance
            solution: Solution to analyze
            filename: Output filename
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Calculate costs per aircraft
        aircraft_ids = []
        early_costs = []
        late_costs = []
        total_costs = []

        for aircraft in instance.aircraft:
            landing_time = solution.get_landing_time(aircraft.id)

            if landing_time < aircraft.target_time:
                early_cost = aircraft.early_penalty * (aircraft.target_time - landing_time)
                late_cost = 0
            else:
                early_cost = 0
                late_cost = aircraft.late_penalty * (landing_time - aircraft.target_time)

            aircraft_ids.append(f'A{aircraft.id}')
            early_costs.append(early_cost)
            late_costs.append(late_cost)
            total_costs.append(early_cost + late_cost)

        # Plot 1: Stacked bar chart
        x = np.arange(len(aircraft_ids))
        width = 0.6

        ax1.bar(x, early_costs, width, label='Early Cost',
               color='skyblue', edgecolor='black', linewidth=0.5)
        ax1.bar(x, late_costs, width, bottom=early_costs,
               label='Late Cost', color='salmon',
               edgecolor='black', linewidth=0.5)

        ax1.set_xlabel('Aircraft', fontweight='bold')
        ax1.set_ylabel('Cost', fontweight='bold')
        ax1.set_title('Cost Breakdown by Aircraft', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(aircraft_ids, rotation=45)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Plot 2: Time deviations
        deviations = []
        for aircraft in instance.aircraft:
            landing_time = solution.get_landing_time(aircraft.id)
            deviation = landing_time - aircraft.target_time
            deviations.append(deviation)

        colors = ['skyblue' if d < 0 else 'salmon' for d in deviations]

        ax2.bar(x, deviations, width, color=colors,
               edgecolor='black', linewidth=0.5)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        ax2.set_xlabel('Aircraft', fontweight='bold')
        ax2.set_ylabel('Time Deviation from Target', fontweight='bold')
        ax2.set_title('Landing Time Deviations', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(aircraft_ids, rotation=45)
        ax2.grid(axis='y', alpha=0.3)

        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor='skyblue', edgecolor='black',
                          label='Early (< Target)'),
            mpatches.Patch(facecolor='salmon', edgecolor='black',
                          label='Late (> Target)')
        ]
        ax2.legend(handles=legend_elements)

        plt.tight_layout()

        if filename:
            filepath = self.output_dir / f"{filename}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Cost breakdown saved: {filepath}")
        else:
            plt.show()

        plt.close()

    def plot_comparison(
        self,
        instance: ProblemInstance,
        heuristic_solution: Solution,
        optimal_solution: Solution,
        filename: Optional[str] = None
    ):
        """
        Create side-by-side comparison of heuristic vs optimal.

        Args:
            instance: Problem instance
            heuristic_solution: Heuristic solution
            optimal_solution: Optimal solution
            filename: Output filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # 1. Cost comparison (top-left)
        ax = axes[0, 0]
        methods = ['Heuristic', 'Optimal']
        costs = [heuristic_solution.objective_value,
                optimal_solution.objective_value]

        bars = ax.bar(methods, costs, color=['skyblue', 'lightgreen'],
                     edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar, cost in zip(bars, costs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{cost:.2f}',
                   ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Total Cost', fontweight='bold')
        ax.set_title('Solution Cost Comparison', fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3)

        # Add gap percentage
        gap = ((heuristic_solution.objective_value - optimal_solution.objective_value) /
               optimal_solution.objective_value * 100)
        ax.text(0.5, 0.95, f'Gap: {gap:.2f}%',
               transform=ax.transAxes,
               ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
               fontsize=11, fontweight='bold')

        # 2. Time comparison (top-right)
        ax = axes[0, 1]
        times = [heuristic_solution.solve_time,
                optimal_solution.solve_time]

        bars = ax.bar(methods, times, color=['skyblue', 'lightgreen'],
                     edgecolor='black', linewidth=1.5)

        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time:.3f}s',
                   ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Solve Time (seconds)', fontweight='bold')
        ax.set_title('Computation Time Comparison', fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3)

        # Add speedup
        speedup = optimal_solution.solve_time / heuristic_solution.solve_time
        ax.text(0.5, 0.95, f'Speedup: {speedup:.1f}x',
               transform=ax.transAxes,
               ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
               fontsize=11, fontweight='bold')

        # 3. Landing time comparison (bottom-left)
        ax = axes[1, 0]
        aircraft_ids = [a.id for a in instance.aircraft]
        x = np.arange(len(aircraft_ids))
        width = 0.35

        heur_times = [heuristic_solution.get_landing_time(aid) for aid in aircraft_ids]
        opt_times = [optimal_solution.get_landing_time(aid) for aid in aircraft_ids]

        ax.bar(x - width/2, heur_times, width, label='Heuristic',
              color='skyblue', edgecolor='black', linewidth=0.5)
        ax.bar(x + width/2, opt_times, width, label='Optimal',
              color='lightgreen', edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Aircraft', fontweight='bold')
        ax.set_ylabel('Landing Time', fontweight='bold')
        ax.set_title('Landing Times Comparison', fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([f'A{aid}' for aid in aircraft_ids], rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 4. Summary statistics (bottom-right)
        ax = axes[1, 1]
        ax.axis('off')

        # Create summary table
        summary_data = {
            'Metric': [
                'Aircraft',
                'Runways',
                'Heuristic Cost',
                'Optimal Cost',
                'Gap (%)',
                'Heuristic Time (s)',
                'Optimal Time (s)',
                'Speedup'
            ],
            'Value': [
                instance.num_aircraft,
                len(set(heuristic_solution.runway_assignments.values())),
                f'{heuristic_solution.objective_value:.2f}',
                f'{optimal_solution.objective_value:.2f}',
                f'{gap:.2f}',
                f'{heuristic_solution.solve_time:.4f}',
                f'{optimal_solution.solve_time:.4f}',
                f'{speedup:.2f}x'
            ]
        }

        df_summary = pd.DataFrame(summary_data)

        # Plot table
        table = ax.table(cellText=df_summary.values,
                        colLabels=df_summary.columns,
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.6, 0.4])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(len(df_summary.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(df_summary) + 1):
            for j in range(len(df_summary.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        ax.set_title('Summary Statistics', fontweight='bold',
                    fontsize=12, pad=20)

        plt.suptitle(f'Heuristic vs Optimal Comparison\nInstance: {instance.num_aircraft} aircraft',
                    fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if filename:
            filepath = self.output_dir / f"{filename}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved: {filepath}")
        else:
            plt.show()

        plt.close()

    def plot_runway_analysis(
        self,
        df: pd.DataFrame,
        filename: Optional[str] = None
    ):
        """
        Plot results of runway analysis.

        Args:
            df: DataFrame with runway analysis results
            filename: Output filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Cost vs Runways
        ax = axes[0, 0]
        ax.plot(df['Runways'], df['Optimal Cost'],
               marker='o', linewidth=2, markersize=8,
               label='Optimal', color='green')
        ax.plot(df['Runways'], df['Heuristic Cost'],
               marker='s', linewidth=2, markersize=8,
               label='Heuristic', color='blue')

        ax.set_xlabel('Number of Runways', fontweight='bold')
        ax.set_ylabel('Total Cost', fontweight='bold')
        ax.set_title('Cost vs Number of Runways', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Gap vs Runways
        ax = axes[0, 1]
        ax.plot(df['Runways'], df['Gap (%)'],
               marker='o', linewidth=2, markersize=8, color='orange')
        ax.set_xlabel('Number of Runways', fontweight='bold')
        ax.set_ylabel('Gap (%)', fontweight='bold')
        ax.set_title('Solution Gap vs Runways', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)

        # 3. Solve Time vs Runways
        ax = axes[1, 0]
        ax.plot(df['Runways'], df['Optimal Time (s)'],
               marker='o', linewidth=2, markersize=8,
               label='Optimal', color='green')
        ax.plot(df['Runways'], df['Heuristic Time (s)'],
               marker='s', linewidth=2, markersize=8,
               label='Heuristic', color='blue')

        ax.set_xlabel('Number of Runways', fontweight='bold')
        ax.set_ylabel('Solve Time (seconds)', fontweight='bold')
        ax.set_title('Computation Time vs Runways', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # 4. Cost reduction percentage
        ax = axes[1, 1]
        baseline_cost = df.loc[df['Runways'] == 1, 'Optimal Cost'].values[0]
        cost_reduction = ((baseline_cost - df['Optimal Cost']) / baseline_cost * 100)

        ax.bar(df['Runways'], cost_reduction,
              color='steelblue', edgecolor='black', linewidth=1)

        ax.set_xlabel('Number of Runways', fontweight='bold')
        ax.set_ylabel('Cost Reduction (%)', fontweight='bold')
        ax.set_title('Cost Reduction vs Single Runway', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        plt.tight_layout()

        if filename:
            filepath = self.output_dir / f"{filename}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Runway analysis plot saved: {filepath}")
        else:
            plt.show()

        plt.close()


if __name__ == "__main__":
    # Test visualization
    from .data_loader import DataLoader
    from .heuristic import GreedyHeuristic

    print("Testing Visualization Module...")

    # Create sample instance
    instance = DataLoader.create_sample_instance(num_aircraft=8)

    # Solve
    heuristic = GreedyHeuristic(instance, num_runways=2)
    solution = heuristic.solve()

    # Visualize
    viz = ResultVisualizer(output_dir="test_figures")
    viz.plot_gantt_chart(instance, solution,
                        title="Test Landing Schedule")
    viz.plot_cost_breakdown(instance, solution)

    print("Visualization test completed!")
