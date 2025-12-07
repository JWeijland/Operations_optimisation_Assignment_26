#!/usr/bin/env python3
"""
Visualization Module for Sensitivity Analysis
Creates heatmaps and other visualizations for sensitivity analysis results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional


def create_heatmap(
    data: pd.DataFrame,
    metric: str,
    title: str,
    filename: str,
    output_dir: str,
    cmap: str = 'YlOrRd',
    annot: bool = True,
    fmt: str = '.1f',
    figsize: tuple = (10, 6)
):
    """
    Create a heatmap visualization.

    Args:
        data: Pivot table with data (aircraft as rows, runways as columns)
        metric: Name of the metric being visualized
        title: Plot title
        filename: Output filename
        output_dir: Output directory
        cmap: Colormap name
        annot: Whether to annotate cells with values
        fmt: Format string for annotations
        figsize: Figure size (width, height)
    """
    output_path = Path(output_dir) / "heatmaps"
    output_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=figsize)

    # Create heatmap
    ax = sns.heatmap(
        data,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={'label': metric}
    )

    # Formatting
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Number of Runways', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Aircraft', fontsize=12, fontweight='bold')

    # Rotate y-axis labels to be horizontal
    plt.yticks(rotation=0)

    plt.tight_layout()

    # Save
    save_path = output_path / f"{filename}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Heatmap saved: {save_path}")

    plt.close()


def create_optimal_cost_heatmap(df: pd.DataFrame, output_dir: str, num_replications: int = 1):
    """Create heatmap for optimal costs."""
    # Create pivot table
    if num_replications > 1:
        pivot_data = df.groupby(['num_aircraft', 'num_runways'])['optimal_cost'].mean().reset_index()
    else:
        pivot_data = df[['num_aircraft', 'num_runways', 'optimal_cost']]

    pivot = pivot_data.pivot(
        index='num_aircraft',
        columns='num_runways',
        values='optimal_cost'
    )

    create_heatmap(
        data=pivot,
        metric='Optimal Cost (EUR)',
        title='Optimal Landing Cost by Aircraft Count and Runway Count',
        filename='optimal_cost_heatmap',
        output_dir=output_dir,
        cmap='YlOrRd',
        fmt='.0f'
    )


def create_heuristic_gap_heatmap(df: pd.DataFrame, output_dir: str, num_replications: int = 1):
    """Create heatmap for heuristic gap percentages."""
    if num_replications > 1:
        pivot_data = df.groupby(['num_aircraft', 'num_runways'])['gap_percent'].mean().reset_index()
    else:
        pivot_data = df[['num_aircraft', 'num_runways', 'gap_percent']]

    pivot = pivot_data.pivot(
        index='num_aircraft',
        columns='num_runways',
        values='gap_percent'
    )

    create_heatmap(
        data=pivot,
        metric='Gap (%)',
        title='Heuristic Gap (%) by Aircraft Count and Runway Count',
        filename='gap_percent_heatmap',
        output_dir=output_dir,
        cmap='RdYlGn_r',  # Reverse colormap (red = high gap = bad)
        fmt='.2f'
    )


def create_solve_time_heatmap(df: pd.DataFrame, output_dir: str, num_replications: int = 1):
    """Create heatmap for optimal solve times."""
    if num_replications > 1:
        pivot_data = df.groupby(['num_aircraft', 'num_runways'])['optimal_time_s'].mean().reset_index()
    else:
        pivot_data = df[['num_aircraft', 'num_runways', 'optimal_time_s']]

    pivot = pivot_data.pivot(
        index='num_aircraft',
        columns='num_runways',
        values='optimal_time_s'
    )

    create_heatmap(
        data=pivot,
        metric='Solve Time (seconds)',
        title='Optimal Solver Time by Aircraft Count and Runway Count',
        filename='solve_time_heatmap',
        output_dir=output_dir,
        cmap='Blues',
        fmt='.1f'
    )


def create_speedup_heatmap(df: pd.DataFrame, output_dir: str, num_replications: int = 1):
    """Create heatmap for speedup (optimal_time / heuristic_time)."""
    if num_replications > 1:
        pivot_data = df.groupby(['num_aircraft', 'num_runways'])['speedup'].mean().reset_index()
    else:
        pivot_data = df[['num_aircraft', 'num_runways', 'speedup']]

    pivot = pivot_data.pivot(
        index='num_aircraft',
        columns='num_runways',
        values='speedup'
    )

    create_heatmap(
        data=pivot,
        metric='Speedup (x)',
        title='Solver Speedup by Aircraft Count and Runway Count',
        filename='speedup_heatmap',
        output_dir=output_dir,
        cmap='Greens',
        fmt='.0f'
    )


def create_cost_reduction_heatmap(df: pd.DataFrame, output_dir: str, num_replications: int = 1):
    """
    Create heatmap showing cost reduction from adding runways.
    Shows percentage reduction compared to single runway.
    """
    # Get mean costs
    if num_replications > 1:
        pivot_data = df.groupby(['num_aircraft', 'num_runways'])['optimal_cost'].mean().reset_index()
    else:
        pivot_data = df[['num_aircraft', 'num_runways', 'optimal_cost']]

    pivot = pivot_data.pivot(
        index='num_aircraft',
        columns='num_runways',
        values='optimal_cost'
    )

    # Calculate percentage reduction from single runway
    baseline = pivot[1]  # Single runway costs
    reduction = pd.DataFrame(index=pivot.index, columns=pivot.columns)

    for col in pivot.columns:
        reduction[col] = ((baseline - pivot[col]) / baseline * 100)

    create_heatmap(
        data=reduction,
        metric='Cost Reduction (%)',
        title='Cost Reduction (%) Compared to Single Runway',
        filename='cost_reduction_heatmap',
        output_dir=output_dir,
        cmap='RdYlGn',  # Green = good (high reduction)
        fmt='.1f'
    )


def create_comparison_plots(df: pd.DataFrame, output_dir: str):
    """Create comparison line plots."""
    output_path = Path(output_dir) / "plots"
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Cost vs Aircraft Count (separate line per runway count)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Cost vs Aircraft
    ax = axes[0, 0]
    for runway in sorted(df['num_runways'].unique()):
        subset = df[df['num_runways'] == runway].groupby('num_aircraft')['optimal_cost'].mean()
        ax.plot(subset.index, subset.values, marker='o', linewidth=2, label=f'{runway} runway(s)')

    ax.set_xlabel('Number of Aircraft', fontweight='bold')
    ax.set_ylabel('Optimal Cost (EUR)', fontweight='bold')
    ax.set_title('Cost vs Aircraft Count', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Cost vs Runways (separate line per aircraft count)
    ax = axes[0, 1]
    for aircraft in sorted(df['num_aircraft'].unique()):
        subset = df[df['num_aircraft'] == aircraft].groupby('num_runways')['optimal_cost'].mean()
        ax.plot(subset.index, subset.values, marker='s', linewidth=2, label=f'{aircraft} aircraft')

    ax.set_xlabel('Number of Runways', fontweight='bold')
    ax.set_ylabel('Optimal Cost (EUR)', fontweight='bold')
    ax.set_title('Cost vs Runway Count', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Gap vs Aircraft
    ax = axes[1, 0]
    for runway in sorted(df['num_runways'].unique()):
        subset = df[df['num_runways'] == runway].groupby('num_aircraft')['gap_percent'].mean()
        ax.plot(subset.index, subset.values, marker='o', linewidth=2, label=f'{runway} runway(s)')

    ax.set_xlabel('Number of Aircraft', fontweight='bold')
    ax.set_ylabel('Gap (%)', fontweight='bold')
    ax.set_title('Heuristic Gap vs Aircraft Count', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: Solve Time vs Aircraft
    ax = axes[1, 1]
    for runway in sorted(df['num_runways'].unique()):
        subset = df[df['num_runways'] == runway].groupby('num_aircraft')['optimal_time_s'].mean()
        ax.plot(subset.index, subset.values, marker='o', linewidth=2, label=f'{runway} runway(s)')

    ax.set_xlabel('Number of Aircraft', fontweight='bold')
    ax.set_ylabel('Solve Time (s)', fontweight='bold')
    ax.set_title('Solver Time vs Aircraft Count', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle('Sensitivity Analysis - Comparison Plots', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_path / 'comparison_plots.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Comparison plots saved: {save_path}")
    plt.close()


def create_all_heatmaps(df: pd.DataFrame, output_dir: str, num_replications: int = 1):
    """
    Create all heatmaps and visualizations.

    Args:
        df: Results DataFrame
        output_dir: Output directory
        num_replications: Number of replications per scenario
    """
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # Create heatmaps
    print("\nCreating heatmaps...")
    create_optimal_cost_heatmap(df, output_dir, num_replications)
    create_heuristic_gap_heatmap(df, output_dir, num_replications)
    create_solve_time_heatmap(df, output_dir, num_replications)
    create_speedup_heatmap(df, output_dir, num_replications)
    create_cost_reduction_heatmap(df, output_dir, num_replications)

    # Create comparison plots
    print("\nCreating comparison plots...")
    create_comparison_plots(df, output_dir)

    print("\n" + "=" * 70)
    print("VISUALIZATIONS COMPLETED")
    print("=" * 70)
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 70 + "\n")


def create_summary_table(df: pd.DataFrame, output_dir: str):
    """Create a summary table with key statistics."""
    output_path = Path(output_dir) / "tables"
    output_path.mkdir(parents=True, exist_ok=True)

    # Create summary statistics
    summary = df.groupby(['num_aircraft', 'num_runways']).agg({
        'optimal_cost': ['mean', 'std'],
        'gap_percent': ['mean', 'std'],
        'optimal_time_s': ['mean', 'std'],
        'speedup': ['mean', 'std']
    }).round(2)

    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

    # Save to CSV
    save_path = output_path / 'summary_statistics.csv'
    summary.to_csv(save_path)
    print(f"  Summary table saved: {save_path}")

    return summary


if __name__ == "__main__":
    # Example usage
    print("This module is meant to be imported, not run directly.")
    print("Use sensitivity_analysis.py to run experiments and generate visualizations.")
