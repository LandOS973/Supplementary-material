#!/usr/bin/env python3
"""
Run PPO-EDA in no_interact mode for a given config across all QUBO/NK instances.
Results stored under results/config/<ConfigName>/<InstanceName>/no_interact.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import numpy as np
import torch

from main_expe_overall import (
    DEFAULTS,
    _discover_nk_instances,
    _discover_qubo_instances,
    _is_cuda_oom,
    _load_instances,
)
from main_expe_overall import _rank_vs_global_ranking_excluding_ppo
from eda_strategies.FactoryStrategyEA import FactoryStrategyEA
from environment.qubo import get_Score_trajectoriesQUBO_cuda
from environment.nk import get_Score_trajectoriesNK_cuda, getTensorInstances_NK


NO_INTERACT_KERNEL = "no_interact"
INSTANCE_DIR_RE = re.compile(r"^(?P<problem>QUBO|NK)_dim(?P<dim>\d+)_t(?P<t>\d+)$")


def _parse_config_name(config_name: str) -> dict:
    # Parse config_name like: kjsd__advperagentrankweighted__M4__L24__eps0p01__g0p0005__ds0p05__dm0p05
    parts = config_name.split("__")
    out = {
        "kernel": None,
        "advantage": None,
        "M": None,
        "lambda_": None,
        "epsilon_svgd": None,
        "gamma": None,
        "decay_start_ratio": None,
        "decay_min_factor": None,
    }

    def parse_float(token: str):
        token = token.replace("p", ".").replace("m", "-")
        try:
            return float(token)
        except Exception:
            return None

    for p in parts:
        if p.startswith("k"):
            out["kernel"] = p[1:]
        elif p.startswith("adv"):
            out["advantage"] = p[3:]
        elif p.startswith("M"):
            try:
                out["M"] = int(p[1:])
            except Exception:
                pass
        elif p.startswith("L"):
            try:
                out["lambda_"] = int(p[1:])
            except Exception:
                pass
        elif p.startswith("eps"):
            out["epsilon_svgd"] = parse_float(p[3:])
        elif p.startswith("g"):
            out["gamma"] = parse_float(p[1:])
        elif p.startswith("ds"):
            out["decay_start_ratio"] = parse_float(p[2:])
        elif p.startswith("dm"):
            out["decay_min_factor"] = parse_float(p[2:])
    return out


def _run_once_no_interact(
    problem_ctx,
    params: dict,
    device=None,
    nb_restarts=None,
):
    device = device or DEFAULTS["device"]
    nb_restarts = DEFAULTS["nb_restarts"] if nb_restarts is None else int(nb_restarts)

    kernel_config = {"name": NO_INTERACT_KERNEL, "epsilon_svgd": params["epsilon_svgd"], "gamma": params["gamma"]}

    factory = FactoryStrategyEA()
    strategy = factory.createStrategyEA(
        "PPO-EDA",
        problem_ctx["dim"],
        params["lambda_"],
        device,
        problem_ctx["dim_variables"],
        params["M"],
        learning_rate=params["epsilon_svgd"],
        epsilon_svgd=params["epsilon_svgd"],
        enable_visualization=DEFAULTS["visualization"],
        svgd_gamma=params["gamma"],
        decay_start_ratio=params["decay_start_ratio"],
        decay_min_factor=params["decay_min_factor"],
        decay_enabled=True,
        advantage_cfg=params["advantage"],
        kernel_config=kernel_config,
        no_interact=True,
    ).to(device)

    if problem_ctx["type_problem"] == "QUBO":
        list_scores, history = get_Score_trajectoriesQUBO_cuda(
            strategy,
            problem_ctx["dim"],
            DEFAULTS["nb_instances_test"],
            nb_restarts,
            DEFAULTS["budget"],
            params["lambda_"],
            problem_ctx["tensor_Q_test"],
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
            nb_restarts,
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
            nb_restarts,
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
        list_scores.detach().cpu().numpy() if torch.is_tensor(list_scores) else list_scores
    )
    avg_score = float(scores_array.mean())
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
        kernel=NO_INTERACT_KERNEL,
        advantage=params["advantage"],
        M=params["M"],
        lambda_=params["lambda_"],
        epsilon_svgd=params["epsilon_svgd"],
        gamma=params["gamma"],
        decay_start_ratio=params["decay_start_ratio"],
        decay_min_factor=params["decay_min_factor"],
        bandwith_kernel=None,
        no_interact=True,
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


def _save_history_csv(out_dir, problem_name, entry, config_name=None):
    from main_expe_overall import _save_history_csv as _save

    _save(out_dir, problem_name, NO_INTERACT_KERNEL, entry, ranking=None, config_name=config_name)


def main():
    config_name = input("Config name to test (ex: kjsd__advperagentrankweighted__M4__L24__eps0p01__g0p0005__ds0p05__dm0p05): ").strip()
    if not config_name:
        raise SystemExit("Config name is required.")

    params = _parse_config_name(config_name)
    missing = [k for k, v in params.items() if v is None and k != "kernel"]
    if missing:
        raise SystemExit(f"Invalid config_name, missing: {', '.join(missing)}")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_root = os.path.join(repo_root, "results", "config", config_name)
    Path(out_root).mkdir(parents=True, exist_ok=True)

    instances_root = Path(repo_root) / "source_code" / "instances"
    qubo_instances = _discover_qubo_instances(instances_root / "QUBO", DEFAULTS["nb_instances_test"])
    nk_instances = _discover_nk_instances(instances_root / "nk", DEFAULTS["nb_instances_test"])
    instances = qubo_instances + nk_instances
    if not instances:
        raise SystemExit("Aucune instance QUBO/NK compatible avec nb_instances_test.")

    for inst in instances:
        inst_name = f"{inst['name']}_dim{inst['dim']}_t{inst['type_instance']}"
        inst_dir = os.path.join(out_root, inst_name, "no_interact")
        summary_path = os.path.join(inst_dir, "best_summary.txt")
        if os.path.isfile(summary_path):
            print(f"  -> skip {inst_name} (already done)")
            continue

        print(f"  -> run {inst_name}")
        problem_ctx = _load_instances(inst, DEFAULTS["device"])
        nb_restarts = DEFAULTS["nb_restarts"]
        if inst["name"] == "NK" and inst["dim"] >= 256 and inst["type_instance"] >= 8:
            nb_restarts = min(nb_restarts, 3)
            print(f"     [PRE] NK big instance, start nb_restarts={nb_restarts}")

        success = False
        t0 = time.time()
        while nb_restarts > 0 and not success:
            try:
                avg_score, history, meta = _run_once_no_interact(
                    problem_ctx,
                    params,
                    device=DEFAULTS["device"],
                    nb_restarts=nb_restarts,
                )
                success = True
            except (RuntimeError, Exception) as exc:
                if not _is_cuda_oom(exc):
                    raise
                nb_restarts -= 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if nb_restarts > 0:
                    print(f"     [OOM] retry with nb_restarts={nb_restarts}.")
                else:
                    print("     [OOM] nb_restarts=0, skip instance.")

        if not success:
            continue

        dt = time.time() - t0
        print(f"     avg_score={avg_score:.6f} | runtime={dt:.2f}s")
        _save_history_csv(inst_dir, inst["name"], {"history": history, "meta": meta}, config_name=config_name)

    # Build interact vs no_interact summary
    gap_lines = []
    wins_no = 0
    wins_int = 0
    gap_values = []
    ranks_interact = []
    ranks_no = []
    for inst in instances:
        inst_name = f"{inst['name']}_dim{inst['dim']}_t{inst['type_instance']}"
        normal_summary = Path(out_root) / inst_name / "best_summary.txt"
        no_summary = Path(out_root) / inst_name / "no_interact" / "best_summary.txt"
        if not normal_summary.exists() or not no_summary.exists():
            continue
        from main_expe_overall import _parse_summary_config
        normal_cfg = _parse_summary_config(normal_summary)
        no_cfg = _parse_summary_config(no_summary)
        if not normal_cfg or not no_cfg:
            continue
        try:
            normal_score = float(normal_cfg.get("avg_score"))
            no_score = float(no_cfg.get("avg_score"))
        except Exception:
            continue

        def _normalize_score_sign_local(problem_name: str, values):
            if not values:
                return values
            # Always convert to "higher is better"
            if problem_name.upper() in ("QUBO", "UBQP"):
                return [-val for val in values]
            return values

        norm_norm = _normalize_score_sign_local(inst["name"], [normal_score])[0]
        norm_no = _normalize_score_sign_local(inst["name"], [no_score])[0]
        if norm_no == 0:
            continue
        gap_pct = (norm_norm - norm_no) / abs(norm_no) * 100.0
        if gap_pct > 0:
            wins_int += 1
            verdict = "interact better"
        elif gap_pct < 0:
            wins_no += 1
            verdict = "no_interact better"
        else:
            verdict = "tie"
        gap_values.append(gap_pct)
        gap_lines.append(
            f"{inst_name}: interact={norm_norm:.6f}, no_interact={norm_no:.6f}, gap={gap_pct:.2f}% ({verdict})"
        )

        _best_algo, _best_score, rank_int, _n_rank, _pct, _my_cmp, _best_cmp = (
            _rank_vs_global_ranking_excluding_ppo(
                repo_root, inst["name"], inst["dim"], inst["type_instance"], normal_score
            )
        )
        if rank_int is not None:
            ranks_interact.append(rank_int)
        _best_algo, _best_score, rank_no, _n_rank, _pct, _my_cmp, _best_cmp = (
            _rank_vs_global_ranking_excluding_ppo(
                repo_root, inst["name"], inst["dim"], inst["type_instance"], no_score
            )
        )
        if rank_no is not None:
            ranks_no.append(rank_no)

    summary_path = Path(out_root) / "interact_vs_no_interact.txt"
    with open(summary_path, "w") as f:
        f.write("Interact vs No_Interact summary\n")
        f.write(f"wins_interact: {wins_int}\n")
        f.write(f"wins_no_interact: {wins_no}\n")
        mean_gap = sum(gap_values) / len(gap_values) if gap_values else 0.0
        f.write(f"mean_gap: {mean_gap:.2f}%\n")
        mean_rank_interact = float(np.mean(ranks_interact)) if ranks_interact else None
        median_rank_interact = float(np.median(ranks_interact)) if ranks_interact else None
        mean_rank_no = float(np.mean(ranks_no)) if ranks_no else None
        median_rank_no = float(np.median(ranks_no)) if ranks_no else None
        f.write(f"mean_rank_interact: {mean_rank_interact}\n")
        f.write(f"median_rank_interact: {median_rank_interact}\n")
        f.write(f"mean_rank_no_interact: {mean_rank_no}\n")
        f.write(f"median_rank_no_interact: {median_rank_no}\n")
        f.write("per_instance_gap:\n")
        for line in gap_lines:
            f.write(f"{line}\n")
    print(f"[DONE] Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
