import argparse
import itertools
import os
import random
import re
import sys
import time
from pathlib import Path

SOURCE_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SOURCE_CODE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_CODE_DIR)

import numpy as np
import torch
from omegaconf import OmegaConf

from eda_strategies.FactoryStrategyEA import FactoryStrategyEA
from environment.qubo import getTensorInstances_QUBO, get_Score_trajectoriesQUBO_cuda
from environment.blockwise import get_Score_trajectoriesBLOCK_cuda
from environment.nk import getTensorInstances_NK, get_Score_trajectoriesNK_cuda
from utils.main_utils import rank_vs_global_ranking


DECAY_START_RATIO_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8]
DECAY_MIN_FACTOR_GRID = [0.0001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

PROBLEMS = [
    dict(name="QUBO", dim=64, type_instance=1),
    dict(name="QUBO", dim=64, type_instance=2),
    dict(name="QUBO", dim=64, type_instance=3),
    dict(name="QUBO", dim=128, type_instance=0),
    dict(name="QUBO", dim=128, type_instance=5),
    dict(name="QUBO", dim=256, type_instance=1),
    dict(name="QUBO", dim=256, type_instance=2),
    dict(name="QUBO", dim=256, type_instance=3),
    dict(name="QUBO", dim=256, type_instance=4),
    dict(name="NK", dim=64, type_instance=4),
    dict(name="NK", dim=128, type_instance=4),
    dict(name="NK", dim=128, type_instance=8),
    dict(name="NK", dim=256, type_instance=2),
]

DEFAULTS = dict(
    seed=0,
    nb_instances_test=10,
    nb_restarts=10,
    budget=50000,
    visualization=False,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
)


def _load_kernel_config(kernel_name: str, repo_root: str) -> dict:
    kernel_dir = Path(repo_root) / "config" / "kernel"
    kernel_path = kernel_dir / f"{kernel_name}.yaml"
    if not kernel_path.exists():
        available = ", ".join(sorted(p.stem for p in kernel_dir.glob("*.yaml"))) if kernel_dir.exists() else "none"
        raise FileNotFoundError(
            f"Kernel config '{kernel_name}' introuvable dans {kernel_dir}. Kernels disponibles: {available}"
        )
    cfg = OmegaConf.load(str(kernel_path))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True) or {}
    if "name" not in cfg_dict:
        cfg_dict["name"] = kernel_name
    return cfg_dict


def _parse_summary_config(summary_path: Path) -> dict:
    cfg = {}
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                if not value:
                    continue
                lowered = value.lower()
                if lowered in ("none", "null", "n/a"):
                    parsed = None
                elif lowered in ("true", "false"):
                    parsed = lowered == "true"
                else:
                    try:
                        if any(token in value for token in (".", "e", "E")):
                            parsed = float(value)
                        else:
                            parsed = int(value)
                    except ValueError:
                        parsed = value
                cfg[key] = parsed
    except OSError:
        return {}
    return cfg


def _find_best_kernel_summary(summary_dir: str, problem_type: str):
    summary_root = Path(summary_dir)
    if not summary_root.is_dir():
        return None, None

    best_kernel = None
    best_cfg = None
    best_score = None
    maximize = _is_maximization_problem(str(problem_type).upper())
    pattern = f"{str(problem_type).upper()}_*_best_summary.txt"

    for summary_path in sorted(summary_root.glob(pattern)):
        if not summary_path.is_file():
            continue
        cfg = _parse_summary_config(summary_path)
        if not cfg:
            continue
        score = cfg.get("avg_score")
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue

        kernel = cfg.get("kernel")
        if kernel is None:
            stem = summary_path.stem
            prefix = f"{str(problem_type).upper()}_"
            suffix = "_best_summary"
            if stem.startswith(prefix) and stem.endswith(suffix):
                kernel = stem[len(prefix):-len(suffix)]
            else:
                continue

        if best_score is None:
            best_score = score
            best_kernel = str(kernel)
            best_cfg = cfg
            continue

        if maximize:
            is_better = score > best_score
        else:
            is_better = score < best_score
        if is_better:
            best_score = score
            best_kernel = str(kernel)
            best_cfg = cfg

    return best_kernel, best_cfg


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
    script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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


def _discover_instances(repo_root):
    exp_dir = Path(repo_root) / "results" / "experiments"
    if not exp_dir.exists():
        return []
    instances = []
    seen = set()
    for summary_path in exp_dir.glob("*_dim*_t*/*_best_summary.txt"):
        if "decay" in summary_path.parts or "no_interact" in summary_path.parts:
            continue
        parent_name = summary_path.parent.name
        match = re.match(r"^(?P<name>.+)_dim(?P<dim>\d+)_t(?P<t>\d+)$", parent_name)
        if not match:
            continue
        name = match.group("name")
        dim = int(match.group("dim"))
        type_instance = int(match.group("t"))
        key = (name, dim, type_instance)
        if key in seen:
            continue
        seen.add(key)
        instances.append(dict(name=name, dim=dim, type_instance=type_instance))
    return instances


def _run_once(
    problem_ctx,
    kernel_name,
    advantage,
    M,
    lambda_,
    epsilon_svgd,
    gamma,
    decay_start_ratio,
    decay_min_factor,
    bandwith_kernel,
    no_interact,
):
    device = DEFAULTS["device"]

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
        decay_start_ratio=decay_start_ratio,
        decay_min_factor=decay_min_factor,
        decay_enabled=True,
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
        decay_start_ratio=decay_start_ratio,
        decay_min_factor=decay_min_factor,
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
        f.write(f"decay_start_ratio: {meta['decay_start_ratio']}\n")
        f.write(f"decay_min_factor: {meta['decay_min_factor']}\n")
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


def _get_problem_dir(out_dir, problem_name, dim, type_instance):
    return os.path.join(out_dir, f"{problem_name}_dim{dim}_t{type_instance}", "decay")


def _has_decay_results(out_dir, problem_name, dim, type_instance):
    decay_dir = _get_problem_dir(out_dir, problem_name, dim, type_instance)
    if not os.path.isdir(decay_dir):
        return False
    if any(Path(decay_dir).glob("*_best_summary.txt")):
        return True
    return False


def _load_existing_best(out_dir, problem_name, dim, type_instance, kernel_name):
    summary_path = os.path.join(
        out_dir,
        f"{problem_name}_dim{dim}_t{type_instance}",
        "decay",
        f"{problem_name}_{kernel_name}_best_summary.txt",
    )
    if not os.path.isfile(summary_path):
        return None
    try:
        with open(summary_path, "r") as f:
            lines = f.readlines()
        avg_score = None
        for line in lines:
            lowered = line.strip().lower()
            if lowered.startswith("avg_score"):
                try:
                    avg_score = float(line.split(":", 1)[1].strip())
                except Exception:
                    avg_score = None
        if avg_score is None:
            return None
        return {"history": None, "meta": {"avg_score": avg_score}}
    except Exception:
        return None


def _resolve_best_config(best_cfg, kernel_name, repo_root):
    try:
        kernel_cfg = _load_kernel_config(kernel_name, repo_root)
    except Exception:
        kernel_cfg = {}
    epsilon_svgd = best_cfg.get("epsilon_svgd")
    if epsilon_svgd is None:
        epsilon_svgd = kernel_cfg.get("epsilon_svgd")
    if epsilon_svgd is None:
        epsilon_svgd = 0.01
        print(f"[WARN] epsilon_svgd manquant, fallback a {epsilon_svgd}")
    gamma = best_cfg.get("gamma")
    if gamma is None:
        gamma = kernel_cfg.get("gamma")
    if gamma is None:
        gamma = 0.001
        print(f"[WARN] gamma manquant, fallback a {gamma}")
    bandwith_kernel = best_cfg.get("bandwith_kernel")
    if bandwith_kernel is None:
        bandwith_kernel = kernel_cfg.get("bandwith_kernel") or (kernel_cfg.get("params") or {}).get("bandwith_kernel")
    advantage = best_cfg.get("advantage") or "peragentrankweighted"
    M = int(best_cfg.get("m") or best_cfg.get("M") or 1)
    lambda_ = int(best_cfg.get("lambda") or best_cfg.get("lambda_") or 1)
    no_interact = bool(best_cfg.get("no_interact") or False)
    if no_interact:
        print("[WARN] meilleure config indique no_interact=True; force a False pour le decay")
        no_interact = False
    return dict(
        kernel_name=kernel_name,
        advantage=advantage,
        M=M,
        lambda_=lambda_,
        epsilon_svgd=float(epsilon_svgd),
        gamma=float(gamma),
        bandwith_kernel=bandwith_kernel,
        no_interact=no_interact,
    )


def main():
    parser = argparse.ArgumentParser(description="Decay grid experiments (basee sur meilleure config normale).")
    parser.add_argument("--outdir", type=str, default=None, help="Repertoire ou ecrire les CSV et resumes.")
    parser.add_argument(
        "--include-existing-decay",
        action="store_true",
        help="Inclure les instances qui ont deja des resultats decay.",
    )
    args, _ = parser.parse_known_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    outdir = args.outdir or os.path.join(repo_root, "results", "experiments")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    _set_seeds(DEFAULTS["seed"])

    start_all = time.time()
    problems = _discover_instances(repo_root) or PROBLEMS
    if problems != PROBLEMS:
        print(f"[INFO] Instances detectees depuis results/experiments: {len(problems)}")
    for problem in problems:
        problem_ctx = _load_instances(problem, DEFAULTS["device"])
        instance_name = f"{problem_ctx['type_problem']}_dim{problem_ctx['dim']}_t{problem_ctx['type_instance']}"
        summary_dir = os.path.join(repo_root, "results", "experiments", instance_name)
        if not args.include_existing_decay and _has_decay_results(
            outdir,
            problem_ctx["type_problem"],
            problem_ctx["dim"],
            problem_ctx["type_instance"],
        ):
            print(f"[SKIP] {instance_name} deja present dans decay.")
            continue
        best_kernel, best_cfg = _find_best_kernel_summary(summary_dir, problem_ctx["type_problem"])
        if best_kernel is None or best_cfg is None:
            print(f"[WARN] Aucun resume valide dans {summary_dir}. Skip.")
            continue

        best_params = _resolve_best_config(best_cfg, best_kernel, repo_root)
        existing = _load_existing_best(
            outdir,
            problem_ctx["type_problem"],
            problem_ctx["dim"],
            problem_ctx["type_instance"],
            best_kernel,
        )
        best_entry = None
        if existing:
            best_entry = existing

        expanded = list(itertools.product(DECAY_START_RATIO_GRID, DECAY_MIN_FACTOR_GRID))
        total_runs = len(expanded)
        print(
            f"[{problem_ctx['type_problem']} dim={problem_ctx['dim']} t={problem_ctx['type_instance']}] "
            f"decay grid runs: {total_runs} | kernel={best_kernel}"
        )

        for idx, (decay_start_ratio, decay_min_factor) in enumerate(expanded, 1):
            t0 = time.time()
            print(
                f"> Run {idx}/{total_runs} | problem={problem_ctx['type_problem']} t={problem_ctx['type_instance']} | "
                f"kernel={best_kernel} | decay_start_ratio={decay_start_ratio} | decay_min_factor={decay_min_factor}"
            )
            avg_score, history, meta = _run_once(
                problem_ctx,
                best_params["kernel_name"],
                best_params["advantage"],
                best_params["M"],
                best_params["lambda_"],
                best_params["epsilon_svgd"],
                best_params["gamma"],
                decay_start_ratio,
                decay_min_factor,
                best_params["bandwith_kernel"],
                best_params["no_interact"],
            )
            dt = time.time() - t0
            print(f"   -> avg_score={avg_score:.6f} | runtime={dt:.2f}s")
            if best_entry is None or _is_better_score(problem_ctx["type_problem"], avg_score, best_entry["meta"]["avg_score"]):
                best_entry = {"history": history, "meta": meta}
                print("   -> new best for this problem+kernel (decay).")
                ranking = rank_vs_global_ranking(
                    repo_root,
                    problem_ctx["type_problem"],
                    meta["dim"],
                    meta["type_instance"],
                    avg_score,
                )
                problem_dir = _get_problem_dir(
                    outdir,
                    meta["problem"],
                    meta["dim"],
                    meta["type_instance"],
                )
                _save_history_csv(
                    problem_dir,
                    meta["problem"],
                    best_kernel,
                    {"history": history, "meta": meta},
                    ranking=ranking,
                )

    print(f"[DONE] main_expe_decay completed in {time.time() - start_all:.2f}s. Results in {outdir}")


if __name__ == "__main__":
    main()
