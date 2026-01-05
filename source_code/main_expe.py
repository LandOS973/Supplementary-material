#!/usr/bin/env python3
import argparse
import itertools
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from eda_strategies.FactoryStrategyEA import FactoryStrategyEA
from environment.qubo import getTensorInstances_QUBO, get_Score_trajectoriesQUBO_cuda
from environment.nk import getTensorInstances_NK, get_Score_trajectoriesNK_cuda


# ============
#  Grilles
# ============
LEARNING_RATES = [0.001, 0.005, 0.01, 0.02, 0.05]
ALPHA_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]  # svgd_alpha
GAMMA_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0]  # utilisé pour RBF/PK

M_VALUES = [1, 5, 10]
LAMBDA_VALUES = [5, 10]
ADVANTAGES = ["peragentrankweighted", "normalizedfitness"]
KERNELS = ["rbf", "pk", "hk"]

PROBLEMS = [
    dict(name="QUBO", dim=128, type_instance=5),
    dict(name="NK", dim=128, type_instance=4),
]

DEFAULTS = dict(
    seed=0,
    nb_instances_test=10,
    nb_restarts=10,
    budget=10000,
    learning_rate_svgd=0.2,
    visualization=False,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
)


def _set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _load_instances(problem_cfg, device):
    script_dir = os.path.abspath(os.path.dirname(__file__))
    name = problem_cfg["name"]
    dim = int(problem_cfg["dim"])
    type_instance = int(problem_cfg["type_instance"])

    if name == "QUBO":
        instance_path = os.path.join(script_dir, "instances", "QUBO") + os.sep
        tensor_Q_test = getTensorInstances_QUBO(
            instance_path,
            DEFAULTS["nb_instances_test"],
            DEFAULTS["nb_restarts"],
            dim,
            type_instance,
            device,
            "test",
        )
        return dict(
            type_problem="QUBO",
            dim=dim,
            type_instance=type_instance,
            tensor_Q_test=tensor_Q_test,
            dim_variables=None,
            D=None,
            vectorIndex_th=None,
            tensor_matrix_locus=None,
            tensor_matrix_contrib=None,
        )

    if name == "NK":
        D = 2
        vectorIndex = np.zeros((type_instance + 1))
        for vi in range(type_instance + 1):
            vectorIndex[vi] = D ** (type_instance - vi)
        vectorIndex_th = torch.tensor(vectorIndex, dtype=torch.float32).to(device)
        base_path = os.path.join(script_dir, "instances", "nk", str(dim), str(type_instance)) + os.sep
        return dict(
            type_problem="NK",
            dim=dim,
            type_instance=type_instance,
            tensor_Q_test=None,
            dim_variables=None,
            D=D,
            vectorIndex_th=vectorIndex_th,
            tensor_matrix_locus=None,
            tensor_matrix_contrib=None,
            nk_base_path=base_path,
        )

    raise ValueError(f"Unsupported problem {name}")


def _run_once(problem_ctx, kernel_name, advantage, M, lambda_, lr, alpha, gamma):
    device = DEFAULTS["device"]
    learning_rate_svgd = DEFAULTS["learning_rate_svgd"]

    # kernel configuration
    kernel_config = {"name": kernel_name, "learning_rate_svgd": learning_rate_svgd, "alpha": alpha}
    if kernel_name in ("rbf", "pk") and gamma is not None:
        kernel_config["gamma"] = gamma

    factory = FactoryStrategyEA()
    strategy = factory.createStrategyEA(
        "PPO-EDA",
        problem_ctx["dim"],
        lambda_,
        device,
        problem_ctx["dim_variables"],
        M,
        learning_rate=lr,
        learning_rate_svgd=learning_rate_svgd,
        enable_visualization=DEFAULTS["visualization"],
        svgd_alpha=alpha,
        advantage_cfg=advantage,
        kernel_config=kernel_config,
    ).to(device)

    if problem_ctx["type_problem"] == "QUBO":
        list_scores, history = get_Score_trajectoriesQUBO_cuda(
            strategy,
            problem_ctx["dim"],
            DEFAULTS["nb_instances_test"],
            DEFAULTS["nb_restarts"],
            DEFAULTS["budget"],
            lambda_,
            problem_ctx["tensor_Q_test"],
            device,
            False,
            None,
            enable_visualization=False,
            return_history=True,
        )
    else:
        tensor_matrix_locus, tensor_matrix_contrib, tensor_Q_test = getTensorInstances_NK(
            problem_ctx["nk_base_path"],
            DEFAULTS["nb_instances_test"],
            DEFAULTS["nb_restarts"],
            lambda_,
            problem_ctx["dim"],
            problem_ctx["D"],
            problem_ctx["type_instance"],
            device,
        )
        list_scores, history = get_Score_trajectoriesNK_cuda(
            strategy,
            problem_ctx["dim"],
            problem_ctx["type_instance"],
            problem_ctx["D"],
            DEFAULTS["nb_instances_test"],
            DEFAULTS["nb_restarts"],
            DEFAULTS["budget"],
            lambda_,
            problem_ctx["vectorIndex_th"],
            tensor_matrix_locus,
            tensor_matrix_contrib,
            device,
            False,
            None,
            enable_visualization=False,
            return_history=True,
        )

    avg_score = float(np.mean(
        list_scores if isinstance(list_scores, (list, tuple))
        else (list_scores.detach().cpu().numpy() if torch.is_tensor(list_scores) else list_scores)
    ))

    run_meta = dict(
        problem=problem_ctx["type_problem"],
        dim=problem_ctx["dim"],
        type_instance=problem_ctx["type_instance"],
        kernel=kernel_name,
        advantage=advantage,
        M=M,
        lambda_=lambda_,
        learning_rate=lr,
        alpha=alpha,
        gamma=gamma,
        avg_score=avg_score,
    )
    return avg_score, history, run_meta


def _save_history_csv(out_dir, problem_name, kernel_name, entry):
    history = entry["history"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    runtime = history.get("runtime") or list(range(1, len(history.get("best_fitness", [])) + 1))
    rows = zip(
        runtime,
        history.get("best_fitness", []),
        history.get("avg_hamming", []),
        history.get("avg_l1", []),
        history.get("avg_entropy", []),
    )
    csv_path = os.path.join(out_dir, f"{problem_name}_{kernel_name}_best_metrics.csv")
    with open(csv_path, "w") as f:
        f.write("step,best_fitness,avg_hamming,avg_l1,avg_entropy\n")
        for step, bf, ham, l1, ent in rows:
            f.write(f"{step},{bf},{ham},{l1},{ent}\n")

    summary_path = os.path.join(out_dir, f"{problem_name}_{kernel_name}_best_summary.txt")
    meta = entry["meta"]
    with open(summary_path, "w") as f:
        f.write(f"Problem: {problem_name}\n")
        f.write(f"Kernel: {kernel_name}\n")
        f.write(f"Advantage: {meta['advantage']}\n")
        f.write(f"M: {meta['M']}\n")
        f.write(f"lambda: {meta['lambda_']}\n")
        f.write(f"learning_rate: {meta['learning_rate']}\n")
        f.write(f"alpha: {meta['alpha']}\n")
        f.write(f"gamma: {meta['gamma']}\n")
        f.write(f"avg_score: {meta['avg_score']}\n")


def main():
    parser = argparse.ArgumentParser(description="Grid experiments for RL-EDA.")
    parser.add_argument("--outdir", type=str, default=None, help="Répertoire où écrire les CSV et résumés.")
    args, _ = parser.parse_known_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    outdir = args.outdir or os.path.join(repo_root, "results", "experiments")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    _set_seeds(DEFAULTS["seed"])

    best_per_problem_kernel = {}

    start_all = time.time()
    for problem in PROBLEMS:
        problem_ctx = _load_instances(problem, DEFAULTS["device"])

        combos = itertools.product(
            KERNELS,
            ADVANTAGES,
            M_VALUES,
            LAMBDA_VALUES,
            LEARNING_RATES,
            ALPHA_VALUES,
        )
        for kernel_name, advantage, M, lambda_, lr, alpha in combos:
            gamma_list = GAMMA_VALUES if kernel_name in ("rbf", "pk") else [None]
            for gamma in gamma_list:
                avg_score, history, meta = _run_once(
                    problem_ctx, kernel_name, advantage, M, lambda_, lr, alpha, gamma
                )
                key = (problem_ctx["type_problem"], kernel_name)
                current_best = best_per_problem_kernel.get(key)
                if current_best is None or avg_score < current_best["meta"]["avg_score"]:
                    best_per_problem_kernel[key] = {"history": history, "meta": meta}

    for (problem_name, kernel_name), entry in best_per_problem_kernel.items():
        problem_dir = os.path.join(
            outdir,
            f"{problem_name}_dim{entry['meta']['dim']}_t{entry['meta']['type_instance']}",
        )
        _save_history_csv(problem_dir, problem_name, kernel_name, entry)

    print(f"[DONE] main_expe completed in {time.time() - start_all:.2f}s. Results in {outdir}")


if __name__ == "__main__":
    main()
