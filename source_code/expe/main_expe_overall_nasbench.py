"""
Run PPO-EDA grid on NASBench with a single dataset load.
Stores best_metrics.csv + raw_scores.csv under results/config/<ConfigName>/nasbench/.
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
import sys
from pathlib import Path

SOURCE_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SOURCE_CODE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_CODE_DIR)

import numpy as np
import torch

from eda_strategies.FactoryStrategyEA import FactoryStrategyEA
from environment.nasbench import get_Score_trajectories_nasbench_cuda
from problems.nasbench import (
    NASBENCH_DIM,
    NASBENCH_DIM_VARIABLES,
    load_nasbench_objective,
    resolve_nasbench_data_file,
)


DEFAULTS = dict(
    seed=0,
    nb_instances_test=100,
    budget=50000,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
)

DEFAULT_GRIDS = [
    dict(
        kernels=["rbf"],
        advantages=["globalrankweighted"],
        M_values=[20],
        lambda_values=[10],
        epsilon_svgd=[0.08,0.05],
        gamma=[0.005,0.015],
        decay_start_ratio=[0.01],
        decay_min_factor=[0.01],
    )
]


def _set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _slugify(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        s = f"{float(value):.6f}".rstrip("0").rstrip(".")
        if s in ("", "-0"):
            s = "0"
    else:
        s = str(value)
    s = s.strip().replace(".", "p").replace("-", "m").replace("/", "_")
    return s


def _build_config_name(params: dict) -> str:
    parts = [
        f"k{_slugify(params['kernel'])}",
        f"adv{_slugify(params['advantage'])}",
        f"M{_slugify(params['M'])}",
        f"L{_slugify(params['lambda_'])}",
        f"eps{_slugify(params['epsilon_svgd'])}",
        f"g{_slugify(params['gamma'])}",
        f"ds{_slugify(params['decay_start_ratio'])}",
        f"dm{_slugify(params['decay_min_factor'])}",
    ]
    if params.get("bandwith_kernel") is not None:
        parts.append(f"bw{_slugify(params['bandwith_kernel'])}")
    return "__".join(parts)


def _expand_grid(grid: dict):
    kernels = grid.get("kernels", ["rbf"])
    advantages = grid.get("advantages", ["peragentrankweighted"])
    M_values = grid.get("M_values", [1])
    lambda_values = grid.get("lambda_values", [1])
    epsilon_svgd = grid.get("epsilon_svgd", [0.01])
    gamma = grid.get("gamma", [0.001])
    decay_start_ratio = grid.get("decay_start_ratio", [0.8])
    decay_min_factor = grid.get("decay_min_factor", [0.1])
    bandwith_kernel = grid.get("bandwith_kernel", [None])

    for (kernel, advantage, M, lambda_, eps, gam, ds, dm, bw) in itertools.product(
        kernels,
        advantages,
        M_values,
        lambda_values,
        epsilon_svgd,
        gamma,
        decay_start_ratio,
        decay_min_factor,
        bandwith_kernel,
    ):
        params = dict(
            kernel=str(kernel).lower(),
            advantage=str(advantage),
            M=int(M),
            lambda_=int(lambda_),
            epsilon_svgd=float(eps),
            gamma=float(gam),
            decay_start_ratio=float(ds),
            decay_min_factor=float(dm),
            bandwith_kernel=bw,
        )
        cfg_name = _build_config_name(params)
        yield cfg_name, params


def _save_history_csv(out_dir: str, history: dict) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    runtime = history.get("runtime") or list(range(1, len(history.get("best_fitness", [])) + 1))
    rows = zip(
        runtime,
        history.get("best_fitness", []),
        history.get("avg_hamming", []),
        history.get("avg_l1", []),
        history.get("avg_entropy", []),
        history.get("score_mean", []),
        history.get("score_median", []),
        history.get("score_std", []),
        history.get("score_p2", []),
        history.get("score_p5", []),
        history.get("score_p10", []),
        history.get("score_p25", []),
        history.get("score_p50", []),
        history.get("score_p75", []),
        history.get("score_p90", []),
        history.get("score_p95", []),
        history.get("score_p98", []),
    )
    csv_path = os.path.join(out_dir, "best_metrics.csv")
    with open(csv_path, "w") as f:
        f.write(
            "step,best_fitness,avg_hamming,avg_l1,avg_entropy,"
            "mean,median,std,2%,5%,10%,25%,50%,75%,90%,95%,98%\n"
        )
        for (step, bf, ham, l1, ent, mean, median, std, p2, p5, p10, p25, p50, p75, p90, p95, p98) in rows:
            f.write(
                f"{step},{bf},{ham},{l1},{ent},"
                f"{mean},{median},{std},{p2},{p5},{p10},{p25},{p50},{p75},{p90},{p95},{p98}\n"
            )


def _save_raw_scores_csv(out_dir: str, scores_array):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    raw_path = os.path.join(out_dir, "raw_scores.csv")
    with open(raw_path, "w") as f:
        f.write("score\n")
        for val in scores_array:
            f.write(f"{float(val)}\n")


def main():
    parser = argparse.ArgumentParser(description="NASBench grid runner (single dataset load).")
    parser.add_argument("--data_file", type=str, default=None, help="Path to nasbench_full.tfrecord")
    parser.add_argument("--budget", type=int, default=DEFAULTS["budget"])
    parser.add_argument("--nb_instances_test", type=int, default=DEFAULTS["nb_instances_test"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument(
        "--device",
        type=str,
        default=str(DEFAULTS["device"]),
        help="Torch device, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=None,
        help="Override results root (default: <repo>/results/config)",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print configs without running.")
    parser.add_argument("--verbose", action="store_true", help="Enable progress output.")
    args = parser.parse_args()

    script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    out_root = args.out_root or os.path.join(repo_root, "results", "config")
    device = torch.device(args.device)

    data_file = resolve_nasbench_data_file(script_dir, args.data_file)
    objective = load_nasbench_objective(data_file)

    factory = FactoryStrategyEA()

    dim = NASBENCH_DIM
    dim_variables = list(NASBENCH_DIM_VARIABLES)

    for grid in DEFAULT_GRIDS:
        for cfg_name, params in _expand_grid(grid):
            out_dir = os.path.join(out_root, cfg_name, "nasbench")
            if args.dry_run:
                print(f"[DRY] {cfg_name} -> {out_dir}")
                continue

            print(f"=== {cfg_name} ===")
            _set_seeds(args.seed)

            kernel_cfg = {"name": params["kernel"], "epsilon_svgd": params["epsilon_svgd"], "gamma": params["gamma"]}
            if params.get("bandwith_kernel") is not None:
                kernel_cfg["bandwith_kernel"] = params["bandwith_kernel"]

            strategy = factory.createStrategyEA(
                "PPO-EDA",
                dim,
                params["lambda_"],
                device,
                dim_variables,
                params["M"],
                learning_rate=params["epsilon_svgd"],
                epsilon_svgd=params["epsilon_svgd"],
                enable_visualization=False,
                svgd_gamma=params["gamma"],
                decay_start_ratio=params["decay_start_ratio"],
                decay_min_factor=params["decay_min_factor"],
                decay_enabled=True,
                advantage_cfg=params["advantage"],
                kernel_config=kernel_cfg,
                no_interact=False,
                no_repulsion=False,
                is_nk3=False,
            ).to(device)

            scores_array, history = get_Score_trajectories_nasbench_cuda(
                objective,
                strategy,
                args.nb_instances_test,
                args.budget,
                params["lambda_"],
                device,
                verbose=args.verbose,
                name_file=None,
                return_history=True,
            )

            avg_score = float(np.mean(scores_array)) if len(scores_array) else float("nan")
            print(f"avg_score={avg_score}")

            _save_history_csv(out_dir, history)
            _save_raw_scores_csv(out_dir, scores_array)


if __name__ == "__main__":
    main()
