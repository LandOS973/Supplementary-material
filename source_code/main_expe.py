#!/usr/bin/env python3
import argparse
import itertools
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from eda_strategies.FactoryStrategyEA import FactoryStrategyEA
from environment.qubo import getTensorInstances_QUBO, get_Score_trajectoriesQUBO_cuda
from environment.blockwise import get_Score_trajectoriesBLOCK_cuda
from environment.nk import getTensorInstances_NK, get_Score_trajectoriesNK_cuda
from utils.main_utils import rank_vs_global_ranking


# ============
#  Grilles
# ============
EPSILON_SVGD_GRID = [0.007 ,0.01, 0.03, 0.1, 0.5, 0.8]
GAMMA_GRID = [0.00005, 0.0001, 0.0005 ,0.001, 0.01, 0.05]
M_VALUES = [20, 15 ,10, 5, 3, 1]
LAMBDA_VALUES = [7, 10, 15, 20, 25]
ADVANTAGES = ["peragentrankweighted", "normalizedfitness"]
KERNELS = ["rbf", "pk", "hk", "jsd"]
#KERNELS = [ "pk", "hk", "jsd"]
#KERNELS = ["rbf"]
NO_INTERACT_VALUES = [True, False]
NO_INTERACT_KERNEL = "hk"

PROBLEMS = [
    dict(name="QUBO", dim=256, type_instance=0)
]

DEFAULTS = dict(
    seed=0,
    nb_instances_test=10,
    nb_restarts=10,
    budget=10000,
    visualization=False,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
)


def _is_maximization_problem(problem_type: str) -> bool:
    return problem_type in ("NK", "BLOCK")


def _is_better_score(problem_type: str, new_score: float, best_score: float) -> bool:
    if _is_maximization_problem(problem_type):
        return new_score > best_score
    return new_score < best_score


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

    if name == "BLOCK":
        block_size = type_instance
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if dim % block_size != 0:
            raise ValueError(f"dim={dim} must be divisible by block_size={block_size}")
        return dict(
            type_problem="BLOCK",
            dim=dim,
            type_instance=type_instance,
            block_size=block_size,
            tensor_Q_test=None,
            dim_variables=None,
            D=None,
            vectorIndex_th=None,
            tensor_matrix_locus=None,
            tensor_matrix_contrib=None,
        )

    raise ValueError(f"Unsupported problem {name}")


def _run_once(problem_ctx, kernel_name, advantage, M, lambda_, epsilon_svgd, gamma, bandwith_kernel, no_interact):
    device = DEFAULTS["device"]

    # kernel configuration
    kernel_config = {"name": kernel_name, "epsilon_svgd": epsilon_svgd, "gamma": gamma}
    if kernel_name in ("rbf", "pk") and bandwith_kernel is not None:
        kernel_config["bandwith_kernel"] = bandwith_kernel

    factory = FactoryStrategyEA()
    strategy = factory.createStrategyEA(
        "PPO-EDA",
        problem_ctx["dim"],
        lambda_,
        device,
        problem_ctx["dim_variables"],
        M,
        learning_rate=epsilon_svgd,
        epsilon_svgd=epsilon_svgd,
        enable_visualization=DEFAULTS["visualization"],
        svgd_gamma=gamma,
        advantage_cfg=advantage,
        kernel_config=kernel_config,
        no_interact=no_interact,
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
            enable_visualization=False,
            return_history=True,
        )
    elif problem_ctx["type_problem"] == "BLOCK":
        list_scores, history = get_Score_trajectoriesBLOCK_cuda(
            strategy,
            problem_ctx["dim"],
            problem_ctx["block_size"],
            DEFAULTS["nb_instances_test"],
            DEFAULTS["nb_restarts"],
            DEFAULTS["budget"],
            lambda_,
            device,
            False,
            enable_visualization=False,
            return_history=True,
        )
    else:
        total_lambda = strategy.lambda_
        tensor_matrix_locus, tensor_matrix_contrib, tensor_Q_test = getTensorInstances_NK(
            problem_ctx["nk_base_path"],
            DEFAULTS["nb_instances_test"],
            DEFAULTS["nb_restarts"],
            total_lambda,
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
            total_lambda,
            problem_ctx["vectorIndex_th"],
            tensor_matrix_locus,
            tensor_matrix_contrib,
            device,
            False,
            enable_visualization=False,
            return_history=True,
        )

    scores_array = (
        np.asarray(list_scores)
        if isinstance(list_scores, (list, tuple, np.ndarray))
        else (list_scores.detach().cpu().numpy() if torch.is_tensor(list_scores) else np.asarray(list_scores))
    )
    avg_score = float(np.mean(scores_array))
    median_score = float(np.percentile(scores_array, 50))
    std_score = float(np.std(scores_array))
    p2 = float(np.percentile(scores_array, 2))
    p5 = float(np.percentile(scores_array, 5))
    p10 = float(np.percentile(scores_array, 10))
    p25 = float(np.percentile(scores_array, 25))
    p50 = float(np.percentile(scores_array, 50))
    p75 = float(np.percentile(scores_array, 75))
    p90 = float(np.percentile(scores_array, 90))
    p95 = float(np.percentile(scores_array, 95))
    p98 = float(np.percentile(scores_array, 98))

    run_meta = dict(
        problem=problem_ctx["type_problem"],
        dim=problem_ctx["dim"],
        type_instance=problem_ctx["type_instance"],
        kernel=kernel_name,
        advantage=advantage,
        M=M,
        lambda_=lambda_,
        epsilon_svgd=epsilon_svgd,
        gamma=gamma,
        bandwith_kernel=bandwith_kernel,
        no_interact=no_interact,
        avg_score=avg_score,
        median_score=median_score,
        std_score=std_score,
        p2=p2,
        p5=p5,
        p10=p10,
        p25=p25,
        p50=p50,
        p75=p75,
        p90=p90,
        p95=p95,
        p98=p98,
    )
    return avg_score, history, run_meta


def _save_history_csv(out_dir, problem_name, kernel_name, entry, ranking=None):
    history = entry["history"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    runtime = history.get("runtime") or list(range(1, len(history.get("best_fitness", [])) + 1))
    score_mean = history.get("score_mean", [])
    score_median = history.get("score_median", [])
    score_std = history.get("score_std", [])
    score_p2 = history.get("score_p2", [])
    score_p5 = history.get("score_p5", [])
    score_p10 = history.get("score_p10", [])
    score_p25 = history.get("score_p25", [])
    score_p50 = history.get("score_p50", [])
    score_p75 = history.get("score_p75", [])
    score_p90 = history.get("score_p90", [])
    score_p95 = history.get("score_p95", [])
    score_p98 = history.get("score_p98", [])
    rows = zip(
        runtime,
        history.get("best_fitness", []),
        history.get("avg_hamming", []),
        history.get("avg_l1", []),
        history.get("avg_entropy", []),
        score_mean,
        score_median,
        score_std,
        score_p2,
        score_p5,
        score_p10,
        score_p25,
        score_p50,
        score_p75,
        score_p90,
        score_p95,
        score_p98,
    )
    csv_path = os.path.join(out_dir, f"{problem_name}_{kernel_name}_best_metrics.csv")
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

    summary_path = os.path.join(out_dir, f"{problem_name}_{kernel_name}_best_summary.txt")
    meta = entry["meta"]
    with open(summary_path, "w") as f:
        f.write(f"Problem: {problem_name}\n")
        f.write(f"Kernel: {kernel_name}\n")
        f.write(f"Advantage: {meta['advantage']}\n")
        f.write(f"M: {meta['M']}\n")
        f.write(f"lambda: {meta['lambda_']}\n")
        f.write(f"epsilon_svgd: {meta['epsilon_svgd']}\n")
        f.write(f"gamma: {meta['gamma']}\n")
        f.write(f"bandwith_kernel: {meta['bandwith_kernel']}\n")
        f.write(f"no_interact: {meta['no_interact']}\n")
        f.write(f"avg_score: {meta['avg_score']}\n")
        f.write(f"median_score: {meta['median_score']}\n")
        f.write(f"std_score: {meta['std_score']}\n")
        f.write(
            "percentiles: "
            f"2%={meta['p2']}, 5%={meta['p5']}, 10%={meta['p10']}, 25%={meta['p25']}, "
            f"50%={meta['p50']}, 75%={meta['p75']}, 90%={meta['p90']}, 95%={meta['p95']}, 98%={meta['p98']}\n"
        )
        if ranking:
            best_algo, best_score, my_rank, n_rank, my_pct = ranking
            if best_algo is not None and n_rank:
                pct_str = f"{my_pct:.1f}%" if my_pct is not None else "n/a"
                f.write(f"ranking_best_algo: {best_algo}\n")
                f.write(f"ranking_best_score: {best_score}\n")
                f.write(f"ranking_my_rank: {my_rank}/{n_rank} ({pct_str})\n")
        else:
            f.write("ranking: unavailable\n")

def _load_existing_best(out_dir, problem_name, dim, type_instance, kernel_name, no_interact):
    problem_dir = os.path.join(out_dir, f"{problem_name}_dim{dim}_t{type_instance}")
    if no_interact:
        problem_dir = os.path.join(problem_dir, "no_interact")
    summary_path = os.path.join(problem_dir, f"{problem_name}_{kernel_name}_best_summary.txt")
    if not os.path.isfile(summary_path):
        return None
    try:
        with open(summary_path, "r") as f:
            lines = f.readlines()
        avg_score = None
        mode_value = None
        for line in lines:
            lowered = line.strip().lower()
            if lowered.startswith("no_interact"):
                try:
                    mode_value = line.split(":", 1)[1].strip().lower()
                except Exception:
                    mode_value = None
            if lowered.startswith("avg_score"):
                try:
                    avg_score = float(line.split(":", 1)[1].strip())
                except Exception:
                    avg_score = None
        if mode_value is None:
            return None
        parsed_mode = mode_value in ("true", "1", "yes")
        if parsed_mode != bool(no_interact):
            return None
        if avg_score is None:
            return None
        return {"history": None, "meta": {"avg_score": avg_score}}
    except Exception:
        return None


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
        # initialize with existing best (to avoid overwriting better past runs)
        for k in KERNELS:
            for no_interact in NO_INTERACT_VALUES:
                if no_interact and k != NO_INTERACT_KERNEL:
                    continue
                existing = _load_existing_best(
                    outdir,
                    problem_ctx["type_problem"],
                    problem_ctx["dim"],
                    problem_ctx["type_instance"],
                    k,
                    no_interact,
                )
                if existing:
                    best_per_problem_kernel[(problem_ctx["type_problem"], k, no_interact)] = existing

        expanded = []
        for kernel_name in KERNELS:
            epsilon_list = EPSILON_SVGD_GRID
            for advantage, M, lambda_, no_interact in itertools.product(
                ADVANTAGES, M_VALUES, LAMBDA_VALUES, NO_INTERACT_VALUES
            ):
                if no_interact and kernel_name != NO_INTERACT_KERNEL:
                    continue
                gamma_list = GAMMA_GRID if not no_interact else [GAMMA_GRID[0]]
                if lambda_ == 1 and advantage != "normalizedfitness":
                    continue
                if lambda_ != 1 and advantage == "normalizedfitness":
                    continue
                for epsilon_svgd in epsilon_list:
                    for gamma in gamma_list:
                        bandwith_kernel = None
                        expanded.append(
                            (kernel_name, advantage, M, lambda_, epsilon_svgd, gamma, bandwith_kernel, no_interact)
                        )

        total_runs = len(expanded)
        print(
            f"[{problem_ctx['type_problem']} dim={problem_ctx['dim']} t={problem_ctx['type_instance']}] "
            f"total runs: {total_runs}"
        )

        for idx, (kernel_name, advantage, M, lambda_, epsilon_svgd, gamma, bandwith_kernel, no_interact) in enumerate(
            expanded, 1
        ):
            t0 = time.time()
            bandwith_kernel_str = f"{bandwith_kernel}" if bandwith_kernel is not None else "n/a"
            print(
                f"▶ Run {idx}/{total_runs} | problem={problem_ctx['type_problem']} t={problem_ctx['type_instance']} | "
                f"kernel={kernel_name} (bandwith_kernel={bandwith_kernel_str}) | "
                f"adv={advantage} | M={M} | lambda={lambda_} | "
                f"epsilon_svgd={epsilon_svgd} | gamma={gamma} | no_interact={no_interact}"
            )
            avg_score, history, meta = _run_once(
                problem_ctx, kernel_name, advantage, M, lambda_, epsilon_svgd, gamma, bandwith_kernel, no_interact
            )
            dt = time.time() - t0
            print(f"   ↳ avg_score={avg_score:.6f} | runtime={dt:.2f}s")
            key = (problem_ctx["type_problem"], kernel_name, no_interact)
            current_best = best_per_problem_kernel.get(key)
            if current_best is None or _is_better_score(problem_ctx["type_problem"], avg_score, current_best["meta"]["avg_score"]):
                best_per_problem_kernel[key] = {"history": history, "meta": meta}
                print("   ↳ new best for this problem+kernel.")
                ranking = rank_vs_global_ranking(
                    repo_root,
                    problem_ctx["type_problem"],
                    meta["dim"],
                    meta["type_instance"],
                    avg_score,
                )
                problem_dir = os.path.join(
                    outdir,
                    f"{meta['problem']}_dim{meta['dim']}_t{meta['type_instance']}",
                )
                if no_interact:
                    problem_dir = os.path.join(problem_dir, "no_interact")
                _save_history_csv(
                    problem_dir,
                    meta["problem"],
                    kernel_name,
                    {"history": history, "meta": meta},
                    ranking=ranking,
                )

    print(f"[DONE] main_expe completed in {time.time() - start_all:.2f}s. Results in {outdir}")


if __name__ == "__main__":
    main()
