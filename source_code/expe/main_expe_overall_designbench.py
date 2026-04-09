"""
Run PPO-EDA grid on DesignBench with a single task load.
Stores best_metrics.csv + raw_scores.csv under results/config/<ConfigName>/designbench/<TaskName>/.
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
from environment.designbench import get_Score_trajectories_designbench_cuda
from problems.designbench import infer_task_space, load_designbench_task


DEFAULTS = dict(
    seed=0,
    nb_instances_test=1,
    budget=10000,
    task_name="GFP-v0",
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
)

DEFAULT_GRIDS = [
    dict(
        kernels=["rbf"],
        advantages=["globalrankweighted"],
        M_values=[2],
        lambda_values=[100],
        epsilon_svgd=[0.008],
        gamma=[0.015],
        decay_start_ratio=[0.5],
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
    with open(csv_path, "w", encoding="utf-8") as f:
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
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write("score\n")
        for val in scores_array:
            f.write(f"{float(val)}\n")


def main():
    parser = argparse.ArgumentParser(description="DesignBench grid runner (single task load).")
    parser.add_argument("--task_name", type=str, default=DEFAULTS["task_name"], help="DesignBench task (e.g. GFP-v0)")
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
    parser.add_argument(
        "--oracle_batch_size",
        type=int,
        default=2048,
        help="Batch size used for DesignBench oracle calls.",
    )
    parser.add_argument(
        "--reseed_per_config",
        action="store_true",
        help="Reset RNG seed before each config (legacy behavior).",
    )
    args = parser.parse_args()

    script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    out_root = args.out_root or os.path.join(repo_root, "results", "config")
    device = torch.device(args.device)

    if args.nb_instances_test <= 0:
        raise ValueError("nb_instances_test must be >= 1.")

    task = load_designbench_task(args.task_name, use_cuda=torch.cuda.is_available())
    factory = FactoryStrategyEA()
    _set_seeds(args.seed)

    dim, alphabet_size, _, discrete_ok = infer_task_space(task, 237, 20)
    if not discrete_ok:
        raise RuntimeError("DesignBench task appears continuous/non-integer for this discrete runner.")
    dim_variables = [alphabet_size for _ in range(dim)]
    unique_cardinalities = sorted(set(dim_variables))
    print(
        "[Problem] "
        f"task={args.task_name} | "
        f"nb_variables={dim} | "
        f"cardinalites_uniques={unique_cardinalities} | "
        f"nb_cardinalites_differentes={len(unique_cardinalities)}"
    )

    for grid in DEFAULT_GRIDS:
        for cfg_idx, (cfg_name, params) in enumerate(_expand_grid(grid)):
            out_dir = os.path.join(out_root, cfg_name, "designbench", args.task_name)
            if args.dry_run:
                print(f"[DRY] {cfg_name} -> {out_dir}")
                continue

            print(f"=== {cfg_name} ===")
            if args.reseed_per_config:
                _set_seeds(args.seed + cfg_idx)

            kernel_cfg = {
                "name": params["kernel"],
                "epsilon_svgd": params["epsilon_svgd"],
                "gamma": params["gamma"],
            }
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

            scores_array, history = get_Score_trajectories_designbench_cuda(
                task,
                strategy,
                args.nb_instances_test,
                args.budget,
                params["lambda_"],
                device,
                verbose=args.verbose,
                oracle_batch_size=args.oracle_batch_size,
                name_file=None,
                return_history=True,
            )

            avg_score = float(np.mean(scores_array)) if len(scores_array) else float("nan")
            print(f"avg_score={avg_score}")

            _save_history_csv(out_dir, history)
            _save_raw_scores_csv(out_dir, scores_array)


if __name__ == "__main__":
    main()
