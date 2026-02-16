#!/usr/bin/env python3
"""
Sensitivity Analysis Script with Boxplot Visualization
MultiAgentUnivariateEDA on NK Landscapes (Fixed λ)

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
    - NK Landscapes: All 12 combinations (N=64,128,256; K=1,2,4,8)
    - Budget: 10,000 evaluations per config
    - Runs: 10 instances × 10 restarts = 100 parallel runs per config
    - Output: Raw fitness values in CSV + Boxplot visualizations
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
from environment.nk import getTensorInstances_NK, get_Score_trajectoriesNK_cuda


# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

CONFIGURATIONS = [
    {"M": 1,  "lambda_per_agent": 14},
    {"M": 2,  "lambda_per_agent": 14},
    {"M": 4,  "lambda_per_agent": 14},
    {"M": 8,  "lambda_per_agent": 14},
    {"M": 16, "lambda_per_agent": 14},
]

FIXED_HYPERPARAMS = {
    "epsilon_svgd": 0.025,
    "svgd_gamma": 0.007,
    "decay_start_ratio": 0.15,
    "decay_min_factor": 0.01,
    "decay_enabled": True,
    "kernel_config": {"name": "rbf"},
    "advantage_cfg": "normalizedfitness",
}

# NK Landscape Configurations (All 12 combinations)
NK_CONFIGS = [
    {"N": 64,  "K": 1},
    {"N": 64,  "K": 2},
    {"N": 64,  "K": 4},
    {"N": 64,  "K": 8},
    {"N": 128, "K": 1},
    {"N": 128, "K": 2},
    {"N": 128, "K": 4},
    {"N": 128, "K": 8},
    {"N": 256, "K": 1},
    {"N": 256, "K": 2},
    {"N": 256, "K": 4},
    {"N": 256, "K": 8},
]

# Common NK Parameters
NK_PARAMS = {
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


def load_nk_problem(N: int, K: int, M: int, lambda_per_agent: int) -> Tuple:
    """
    Load real NK instances for specific N and K.
    
    Raises:
        FileNotFoundError: If NK instance path not found
        Exception: If getTensorInstances_NK fails
    """
    D = 2  # NK is typically D=2
    type_instance = K
    
    vectorIndex = np.zeros((type_instance + 1))
    for i in range(type_instance + 1):
        vectorIndex[i] = D ** (type_instance - i)
    vectorIndex_th = torch.tensor(vectorIndex, dtype=torch.float32).to(EXPERIMENT_PARAMS["device"])
    
    nk_path = os.path.join(script_dir, "instances", "nk", str(N), str(K)) + os.sep
    
    if not os.path.exists(nk_path):
        raise FileNotFoundError(f"NK instance path not found: {nk_path}")
    
    tensor_matrix_locus, tensor_matrix_contrib, tensor_Q_test = getTensorInstances_NK(
        nk_path,
        NK_PARAMS["num_instances"],
        NK_PARAMS["num_restarts"],
        M * lambda_per_agent,
        N,
        D,
        type_instance,
        EXPERIMENT_PARAMS["device"],
    )
    return tensor_matrix_locus, tensor_matrix_contrib, tensor_Q_test, vectorIndex_th


def create_strategy(N: int, M: int, lambda_per_agent: int, device: str):
    """
    Create a MultiAgentUnivariateEDA strategy.
    
    Raises:
        Exception: If factory or strategy creation fails
    """
    factory = FactoryStrategyEA()
    
    strategy = factory.createStrategyEA(
        "PPO-EDA",
        N=N,
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

def run_sensitivity_analysis(output_dir: str = ".") -> str:
    """
    Run Sensitivity Analysis using get_Score_trajectoriesNK_cuda.
    Collects RAW fitness values (not normalized).
    """
    device = EXPERIMENT_PARAMS["device"]
    nb_instances = NK_PARAMS["num_instances"]
    nb_restarts = NK_PARAMS["num_restarts"]
    total_budget = EXPERIMENT_PARAMS["budget"]
    
    print(f"\n{'='*80}")
    print(f"SENSITIVITY ANALYSIS: MultiAgentUnivariateEDA (Boxplot Visualization)")
    print(f"{'='*80}")
    print(f"\nDevice: {device}")
    print(f"NK Configurations: {len(NK_CONFIGS)}")
    print(f"M/λ Configurations: {len(CONFIGURATIONS)}")
    print(f"Runs per Config: {nb_instances * nb_restarts} (10 instances × 10 restarts)")
    print(f"Total Budget: {total_budget} evaluations")
    
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    # Loop over ALL NK landscape configurations
    for nk_idx, nk_cfg in enumerate(NK_CONFIGS):
        N = nk_cfg["N"]
        K = nk_cfg["K"]
        print(f"\n>>> NK Landscape {nk_idx+1}/{len(NK_CONFIGS)}: N={N}, K={K}")
        
        # Loop over M/λ configurations (Iso-Cost)
        for cfg_idx, config in enumerate(CONFIGURATIONS):
            M = config["M"]
            lambda_per_agent = config["lambda_per_agent"]
            print(f"  [Config {cfg_idx+1}/{len(CONFIGURATIONS)}] M={M}, λ={lambda_per_agent} ...", end=" ")
            
            # Set seed for reproducibility
            set_seed(NK_PARAMS["seed_base"])
            
            # Load NK Tensors
            try:
                tensor_matrix_locus, tensor_matrix_contrib, tensor_Q_test, vectorIndex_th = load_nk_problem(
                    N, K, M, lambda_per_agent
                )
            except FileNotFoundError as e:
                print(f"[SKIP] {e}")
                continue
            
            # Create strategy (errors NOT caught - let them surface)
            strategy = create_strategy(N, M, lambda_per_agent, device)
            
            # VECTORIZED EXECUTION on GPU
            # One call for all 100 runs (10 instances × 10 restarts)
            scores = get_Score_trajectoriesNK_cuda(
                strategy,
                N,
                K,
                2,  # D=2 for NK
                nb_instances,
                nb_restarts,
                total_budget,
                lambda_per_agent,
                vectorIndex_th,
                tensor_matrix_locus,
                tensor_matrix_contrib,
                device,
                verbose=False,
                enable_visualization=False,
                return_history=False,
            )
            
            # Process raw results (length 100)
            # Store one entry per fitness value (100 total)
            for fitness in scores:
                results.append({
                    "nk_n": N,
                    "nk_k": K,
                    "M": M,
                    "fitness": float(fitness),
                })
            
            print(f"✓ {len(scores)} runs collected")
    
    # Save results
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "sensitivity_nk_boxplot_results.csv")
    results_df.to_csv(csv_path, index=False)
    
    print(f"\n✓ Saved {len(results)} fitness values to {csv_path}")
    print(f"  Columns: {list(results_df.columns)}")
    
    return csv_path


# ============================================================================
# VISUALIZATION (BOXPLOTS)
# ============================================================================

def plot_boxplot_results(csv_path: str, output_dir: Optional[str] = None) -> str:
    """
    Create BOXPLOT visualization per NK configuration.
    
    X-axis: M (Number of Agents)
    Y-axis: Raw Best Fitness
    
    Args:
        csv_path: Path to CSV file with results
        output_dir: Directory to save plots (defaults to same as CSV)
    
    Returns:
        Path to plots directory
    """
    if output_dir is None:
        output_dir = os.path.dirname(csv_path) or "."
    
    print(f"\n{'='*80}")
    print("VISUALIZATION: Boxplot Results (Raw Fitness)")
    print(f"{'='*80}")
    
    # Load data
    results_df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(results_df)} fitness values from {csv_path}")
    
    # Create subdirectory for plots
    plots_dir = os.path.join(output_dir, "boxplots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get unique NK configurations (sorted)
    nk_pairs = sorted(results_df[["nk_n", "nk_k"]].drop_duplicates().values.tolist())
    print(f"Found {len(nk_pairs)} NK configurations to plot")
    
    # Create a boxplot for each NK configuration
    for nk_n, nk_k in nk_pairs:
        nk_data = results_df[(results_df["nk_n"] == nk_n) & (results_df["nk_k"] == nk_k)]
        
        print(f"\nPlotting NK({nk_n}, {nk_k}) - {len(nk_data)} fitness values")
        
        # Create boxplot
        fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
        
        # Prepare data for boxplot (grouped by M)
        sns.boxplot(
            data=nk_data,
            x="M",
            y="fitness",
            palette="viridis",
            ax=ax,
            width=0.6,
            showfliers=False
        )
        
        # Formatting
        ax.set_xlabel("Number of Agents (M)", fontsize=13, fontweight='bold')
        ax.set_ylabel("Best Fitness (Raw)", fontsize=13, fontweight='bold')
        ax.set_title(f"NK(N={nk_n}, K={nk_k})", 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Grid for better readability
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(plots_dir, f"boxplot_NK_{nk_n}_{nk_k}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {plot_path}")
        plt.close()
    
    print(f"\n✓ All {len(nk_pairs)} boxplots saved to {plots_dir}")
    return plots_dir


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def print_summary_statistics(csv_path: str):
    """Print summary statistics per NK configuration and M."""
    results_df = pd.read_csv(csv_path)
    
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS (Per NK Configuration and M)")
    print(f"{'='*80}\n")
    
    for nk_n, nk_k in sorted(results_df[["nk_n", "nk_k"]].drop_duplicates().values.tolist()):
        nk_data = results_df[(results_df["nk_n"] == nk_n) & (results_df["nk_k"] == nk_k)]
        
        print(f"NK(N={nk_n}, K={nk_k}):")
        
        summary = nk_data.groupby("M")["fitness"].agg([
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
    output_dir = os.path.join(script_dir, "sensitivity_analysis_boxplot")
    
    # Run sensitivity analysis (collect raw fitness values)
    csv_path = run_sensitivity_analysis(output_dir=output_dir)
    
    # Print summary statistics
    print_summary_statistics(csv_path)
    
    # Create boxplot visualizations
    plots_dir = plot_boxplot_results(csv_path, output_dir=output_dir)
    
    print(f"\n{'='*80}")
    print("SENSITIVITY ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults Summary:")
    print(f"  • CSV File: {csv_path}")
    print(f"  • Plots Directory: {plots_dir}")
    print(f"\nNext Steps:")
    print(f"  1. Review CSV for raw fitness values")
    print(f"  2. Inspect boxplots to assess M impact on fitness distribution")
    print(f"  3. Analyze stability and outliers per configuration")


if __name__ == "__main__":
    main()
