#!/usr/bin/env python3
"""
Sensitivity Analysis Script with Boxplot Visualization
MultiAgentUnivariateEDA on QUBO Problems (Fixed λ, Minimization)

Goal:
    Analyze the impact of the number of agents M on final performance under fixed budget.
    Use BOXPLOTS to visualize distribution, stability, and outliers of results.

Experimental Setup (Fixed λ = 14):
    Population per agent: λ = 14 (constant across all M)
    
    Configurations:
    - M=1   (1 agent × 14 population)
    - M=2   (2 agents × 14 population)
    - M=4   (4 agents × 14 population)
    - M=8   (8 agents × 14 population)
    - M=16  (16 agents × 14 population)

Fixed Hyperparameters:
    - epsilon_svgd: 0.025
    - svgd_gamma: 0.007
    - decay_start_ratio: 0.15
    - decay_min_factor: 0.01
    - decay_enabled: True
    - kernel_config: {"name": "rbf"}
    - advantage_cfg: "baseline"

Experimental Protocol:
    - QUBO Problems: Multiple dimensions (N=64, N=128, etc.)
    - Budget: 10,000 evaluations per config
    - Runs: 10 instances × 10 restarts = 100 parallel runs per config
    - Output: Raw fitness values in CSV + Boxplot visualizations
    - Objective: Minimization (lower fitness = better)
"""

import os
import sys
import random
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Add source_code directory to path
script_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(script_dir))

# Import from local codebase (NO try...except - let errors surface)
from eda_strategies.FactoryStrategyEA import FactoryStrategyEA
from environment.qubo import getTensorInstances_QUBO, get_Score_trajectoriesQUBO_cuda


# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================
CONFIGURATIONS = [
    {"M": 1,  "lambda_per_agent": 10},
    {"M": 2,  "lambda_per_agent": 10},
    {"M": 4,  "lambda_per_agent": 10},
    {"M": 8,  "lambda_per_agent": 10},
    {"M": 10, "lambda_per_agent": 10},
]

FIXED_HYPERPARAMS = {
    "epsilon_svgd": 0.015,
    "svgd_gamma": 0.005,
    "decay_start_ratio": 0.01,
    "decay_min_factor": 0.01,
    "decay_enabled": True,
    "kernel_config": {"name": "rbf"},
    "advantage_cfg": "normalizedfitness",
}
# QUBO Problem Configurations
# All combinations of N ∈ {64, 128, 256} and Type ∈ {0, 1, 2, 3, 4, 5}
QUBO_CONFIGS = [
    {"n": 64,  "type_instance": 0},
    {"n": 64,  "type_instance": 1},
    {"n": 64,  "type_instance": 2},
    {"n": 64,  "type_instance": 3},
    {"n": 64,  "type_instance": 4},
    {"n": 64,  "type_instance": 5},
    {"n": 128, "type_instance": 0},
    {"n": 128, "type_instance": 1},
    {"n": 128, "type_instance": 2},
    {"n": 128, "type_instance": 3},
    {"n": 128, "type_instance": 4},
    {"n": 128, "type_instance": 5},
    {"n": 256, "type_instance": 0},
    {"n": 256, "type_instance": 1},
    {"n": 256, "type_instance": 2},
    {"n": 256, "type_instance": 3},
    {"n": 256, "type_instance": 4},
    {"n": 256, "type_instance": 5},
]

# Common QUBO Parameters
QUBO_PARAMS = {
    "num_instances": 10,
    "num_restarts": 10,
    "seed_base": 42,
}

# Experiment Parameters
EXPERIMENT_PARAMS = {
    "budget": 10000,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def load_qubo_problem(
    n: int,
    type_instance: int,
    M: int,
    lambda_per_agent: int,
    num_restarts: Optional[int] = None,
) -> Tuple:
    """
    Load real QUBO instances for specific dimension n and instance type.
    
    Args:
        n: Problem dimension
        type_instance: Instance type (0-5)
        M: Number of agents
        lambda_per_agent: Population per agent
    
    Raises:
        Exception: If getTensorInstances_QUBO fails
    """
    instance_path = os.path.join(script_dir, "instances", "QUBO") + os.sep
    
    if num_restarts is None:
        num_restarts = QUBO_PARAMS["num_restarts"]

    tensor_Q = getTensorInstances_QUBO(
        instance_path,
        QUBO_PARAMS["num_instances"],
        num_restarts,
        n,
        type_instance,
        EXPERIMENT_PARAMS["device"],
        "test",
    )
    return tensor_Q


def create_strategy(n: int, M: int, lambda_per_agent: int, device: str):
    """
    Create a MultiAgentUnivariateEDA strategy for QUBO (Minimization).
    
    Raises:
        Exception: If factory or strategy creation fails
    """
    factory = FactoryStrategyEA()
    
    strategy = factory.createStrategyEA(
        "PPO-EDA",
        N=n,
        lambda_=lambda_per_agent,
        device=device,
        dim_variables=None,
        M=M,
        learning_rate=FIXED_HYPERPARAMS["epsilon_svgd"],
        epsilon_svgd=FIXED_HYPERPARAMS["epsilon_svgd"],
        enable_visualization=False,
        svgd_gamma=FIXED_HYPERPARAMS["svgd_gamma"],
        decay_start_ratio=FIXED_HYPERPARAMS["decay_start_ratio"],
        decay_min_factor=FIXED_HYPERPARAMS["decay_min_factor"],
        decay_enabled=FIXED_HYPERPARAMS["decay_enabled"],
        advantage_cfg=FIXED_HYPERPARAMS["advantage_cfg"],
        kernel_config=FIXED_HYPERPARAMS["kernel_config"],
        no_interact=False,
        no_repulsion=False,
    ).to(device)
    return strategy


# ============================================================================
# MAIN SENSITIVITY ANALYSIS
# ============================================================================

def run_sensitivity_analysis(output_dir: str = ".") -> Tuple[str, str]:
    """
    Run Sensitivity Analysis using get_Score_trajectoriesQUBO_cuda.
    Collects RAW fitness values (not normalized).
    QUBO is a minimization problem (lower is better).
    """
    device = EXPERIMENT_PARAMS["device"]
    nb_instances = QUBO_PARAMS["num_instances"]
    nb_restarts = QUBO_PARAMS["num_restarts"]
    total_budget = EXPERIMENT_PARAMS["budget"]
    
    print(f"\n{'='*80}")
    print(f"SENSITIVITY ANALYSIS: MultiAgentUnivariateEDA (QUBO - Boxplot Visualization)")
    print(f"{'='*80}")
    print(f"\nDevice: {device}")
    print(f"QUBO Configurations: {len(QUBO_CONFIGS)} (N × Type combinations)")
    print(f"M/λ Configurations: {len(CONFIGURATIONS)}")
    print(f"Runs per Config: {nb_instances * nb_restarts} (10 instances × 10 restarts)")
    print(f"Total Budget: {total_budget} evaluations")
    print(f"Objective: Minimization (lower fitness = better)")
    
    os.makedirs(output_dir, exist_ok=True)
    results = []
    history_results = []
    
    # Loop over ALL QUBO problem configurations (N × Type)
    for qubo_idx, qubo_cfg in enumerate(QUBO_CONFIGS):
        n = qubo_cfg["n"]
        type_instance = qubo_cfg["type_instance"]
        print(f"\n>>> QUBO Problem {qubo_idx+1}/{len(QUBO_CONFIGS)}: N={n}, Type={type_instance}")
        
        # Loop over M/λ configurations (Iso-Cost)
        for cfg_idx, config in enumerate(CONFIGURATIONS):
            M = config["M"]
            lambda_per_agent = config["lambda_per_agent"]
            print(f"  [Config {cfg_idx+1}/{len(CONFIGURATIONS)}] M={M}, λ={lambda_per_agent} ...", end=" ")
            
            # Set seed for reproducibility
            set_seed(QUBO_PARAMS["seed_base"])
            
            device_for_config = device
            restarts_for_config = nb_restarts
            # Load QUBO Tensors (reduce restarts on OOM)
            while True:
                try:
                    tensor_Q = load_qubo_problem(
                        n, type_instance, M, lambda_per_agent, num_restarts=restarts_for_config
                    )
                    break
                except torch.OutOfMemoryError:
                    if not torch.cuda.is_available():
                        raise
                    if restarts_for_config <= 1:
                        raise
                    torch.cuda.empty_cache()
                    new_restarts = max(1, restarts_for_config // 2)
                    print(
                        f"[OOM] Reducing restarts from {restarts_for_config} to {new_restarts} "
                        f"and retrying..."
                    )
                    restarts_for_config = new_restarts
            
            # Create strategy (errors NOT caught - let them surface)
            strategy = create_strategy(n, M, lambda_per_agent, device_for_config)
            
            # VECTORIZED EXECUTION on GPU
            # One call for all 100 runs (10 instances × 10 restarts)
            while True:
                try:
                    scores, history = get_Score_trajectoriesQUBO_cuda(
                        strategy,
                        n,
                        nb_instances,
                        restarts_for_config,
                        total_budget,
                        lambda_per_agent,
                        tensor_Q,
                        device_for_config,
                        verbose=False,
                        enable_visualization=False,
                        return_history=True,
                    )
                    break
                except torch.OutOfMemoryError as e:
                    if not torch.cuda.is_available():
                        raise
                    if restarts_for_config <= 1:
                        raise
                    torch.cuda.empty_cache()
                    new_restarts = max(1, restarts_for_config // 2)
                    print(
                        f"[OOM] Reducing restarts from {restarts_for_config} to {new_restarts} "
                        f"and retrying..."
                    )
                    restarts_for_config = new_restarts
                    # Reload QUBO tensors with the reduced number of restarts
                    tensor_Q = load_qubo_problem(
                        n, type_instance, M, lambda_per_agent, num_restarts=restarts_for_config
                    )
                    # Recreate strategy to reset internal buffers sized by batch
                    strategy = create_strategy(n, M, lambda_per_agent, device_for_config)
            
            # Process raw results (length 100)
            # Store one entry per fitness value (100 total)
            for fitness in scores:
                results.append({
                    "problem_type": "QUBO",
                    "n": n,
                    "type_instance": type_instance,
                    "M": M,
                    "fitness": float(fitness),
                })
            
            # Curves disabled
            
            print(f"✓ {len(scores)} runs collected")
    
    # Save results
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "sensitivity_qubo_boxplot_results.csv")
    results_df.to_csv(csv_path, index=False)
    
    print(f"\n✓ Saved {len(results)} fitness values to {csv_path}")
    
    return csv_path, None


# ============================================================================
# VISUALIZATION (BOXPLOTS & CURVES)
# ============================================================================

def plot_boxplot_results(csv_path: str, output_dir: Optional[str] = None) -> str:
    """
    Create BOXPLOT visualization per QUBO dimension.
    
    X-axis: M (Number of Agents)
    Y-axis: Raw Best Fitness (Minimization: lower is better)
    
    Args:
        csv_path: Path to CSV file with results
        output_dir: Directory to save plots (defaults to same as CSV)
    
    Returns:
        Path to plots directory
    """
    if output_dir is None:
        output_dir = os.path.dirname(csv_path) or "."
    
    print(f"\n{'='*80}")
    print("VISUALIZATION: Boxplot Results (Raw Fitness - Minimization)")
    print(f"{'='*80}")
    
    # Load data
    results_df = pd.read_csv(csv_path)
    # QUBO: display scores in positive (maximization-style)
    results_df["fitness"] = results_df["fitness"].abs()
    print(f"\nLoaded {len(results_df)} fitness values from {csv_path}")
    
    # Create subdirectory for plots
    plots_dir = os.path.join(output_dir, "boxplots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get unique QUBO (N, Type) pairs (sorted)
    qubo_pairs = sorted(results_df[["n", "type_instance"]].drop_duplicates().values.tolist())
    print(f"Found {len(qubo_pairs)} QUBO (N, Type) configurations to plot")
    
    # Create a boxplot for each (N, Type) pair
    for n, type_instance in qubo_pairs:
        qubo_data = results_df[(results_df["n"] == n) & (results_df["type_instance"] == type_instance)]
        
        print(f"\nPlotting QUBO(N={n}, Type={type_instance}) - {len(qubo_data)} fitness values")
        
        # Create boxplot
        fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
        
        # Prepare data for boxplot (grouped by M)
        sns.boxplot(
            data=qubo_data,
            x="M",
            y="fitness",
            palette="viridis",
            ax=ax,
            width=0.6,
            showfliers=False
        )
        
        # Formatting
        ax.set_xlabel("Number of Agents (m)", fontsize=20, fontweight='bold')
        ax.set_ylabel("Fitness", fontsize=20, fontweight='bold')
        ax.set_title(f"QUBO (N={n}, Type={type_instance})", 
                     fontsize=18, fontweight='bold', pad=20)
        ax.tick_params(axis="both", labelsize=16)
        
        # Grid for better readability
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(plots_dir, f"boxplot_QUBO_{n}_type{type_instance}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {plot_path}")
        plt.close()
    
    print(f"\n✓ All {len(qubo_pairs)} boxplots saved to {plots_dir}")
    return plots_dir


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def print_summary_statistics(csv_path: str):
    """Print summary statistics per QUBO (N, Type) configuration and M."""
    results_df = pd.read_csv(csv_path)
    # QUBO: display scores in positive (maximization-style)
    results_df["fitness"] = results_df["fitness"].abs()
    
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS (Per QUBO Configuration and M)")
    print(f"{'='*80}")
    print("(Minimization: Lower fitness is better)\n")
    
    qubo_pairs = sorted(results_df[["n", "type_instance"]].drop_duplicates().values.tolist())
    for n, type_instance in qubo_pairs:
        qubo_data = results_df[(results_df["n"] == n) & (results_df["type_instance"] == type_instance)]
        
        print(f"QUBO(N={n}, Type={type_instance}):")
        
        summary = qubo_data.groupby("M")["fitness"].agg([
            ("Count", "count"),
            ("Mean", "mean"),
            ("Std", "std"),
            ("Min", "min"),
            ("Q1", lambda x: x.quantile(0.25)),
            ("Median", "median"),
            ("Q3", lambda x: x.quantile(0.75)),
            ("Max", "max"),
        ]).round(6)
        
        print(summary.to_string())
        print()


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point for sensitivity analysis."""
    output_dir = os.path.join(script_dir, "sensitivity_analysis_qubo_boxplot")
    csv_path = os.path.join(output_dir, "sensitivity_qubo_boxplot_results.csv")
    
    # Run sensitivity analysis only if results are not already saved
    if os.path.exists(csv_path):
        print(f"[INFO] Using existing results CSV: {csv_path}")
    else:
        csv_path, _ = run_sensitivity_analysis(output_dir=output_dir)
    
    # Print summary statistics
    print_summary_statistics(csv_path)
    
    # Create boxplot visualizations
    plots_dir = plot_boxplot_results(csv_path, output_dir=output_dir)
    
    print(f"\n{'='*80}")
    print("SENSITIVITY ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults Summary:")
    print(f"  • CSV File: {csv_path}")
    print(f"  • Boxplots Directory: {plots_dir}")
    print(f"\nNext Steps:")
    print(f"  1. Review CSV for raw fitness values (minimization: lower is better)")
    print(f"  2. Inspect boxplots to assess M impact on fitness distribution")


if __name__ == "__main__":
    main()
