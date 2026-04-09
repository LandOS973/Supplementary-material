"""
Hydra runner for Nevergrad DiscreteDE on ViennaRNA inverse folding.

Objective used:
score = -ensemble_defect(target)

Nevergrad minimizes by default, so we optimize loss = -score.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from problems.viennarna import (
    ETERNA100_TSV_URL,
    RNA_ALPHABET,
    load_target_from_eterna100,
    normalize_target_struct,
)

try:
    import nevergrad as ng
except Exception:                    
    ng = None

try:
    import RNA
except Exception:                    
    RNA = None


DEFAULT_TARGET_NAME = "Simple Hairpin"
DEFAULT_TARGET_STRUCT = "(" * 25 + "." * 50 + ")" * 25


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
    return s.strip().replace(".", "p").replace("-", "m").replace("/", "_").replace(" ", "_")


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


def _save_raw_scores_csv(out_dir: str, scores_array) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    raw_path = os.path.join(out_dir, "raw_scores.csv")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write("score\n")
        for val in scores_array:
            f.write(f"{float(val)}\n")


def _score_tokens(tokens: np.ndarray, target_struct: str) -> float:
    seq = "".join(RNA_ALPHABET[np.asarray(tokens, dtype=np.int64)])
    fc = RNA.fold_compound(seq)
    fc.pf()
    defect = float(fc.ensemble_defect(target_struct))
    return -defect


def _run_restart(
    target_struct: str,
    dim: int,
    budget: int,
    step_record: int,
    algo_name: str,
    seed: int,
    progress_bar=None,
) -> tuple[float, list[float]]:
    param = ng.p.TransitionChoice(range(4), repetitions=dim, ordered=False)
    algo_cls = ng.optimizers.registry.get(algo_name)
    optimizer = algo_cls(parametrization=param, budget=budget)
    optimizer.parametrization.random_state.seed(seed)

    best_score = -float("inf")
    checkpoints = []
    for step in range(1, budget + 1):
        candidate = optimizer.ask()
        score = _score_tokens(np.asarray(candidate.value), target_struct)
        loss = -score
        optimizer.tell(candidate, loss)
        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix(current_score=float(score), best_score=float(max(best_score, score)))
        if score > best_score:
            best_score = score
        if step % step_record == 0:
            checkpoints.append(best_score)
    if budget % step_record != 0:
        checkpoints.append(best_score)
    return best_score, checkpoints


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if RNA is None:
        raise RuntimeError(
            "Import `RNA` failed. Install ViennaRNA Python bindings first "
            "(e.g. `pip install ViennaRNA`)."
        )
    if ng is None:
        raise RuntimeError(
            "Import `nevergrad` failed. Install Nevergrad first "
            "(e.g. `pip install nevergrad`)."
        )

    script_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))

    target_name = str(OmegaConf.select(cfg, "problem.target_name") or DEFAULT_TARGET_NAME)
    target_source = str(OmegaConf.select(cfg, "problem.target_source") or ETERNA100_TSV_URL)
    target_struct_cfg = OmegaConf.select(cfg, "problem.target_struct")
    if target_struct_cfg:
        target_struct = normalize_target_struct(str(target_struct_cfg))
        target_resolved_name = "cfg_target_struct"
        print(f"[ViennaRNA] using target from cfg.problem.target_struct (len={len(target_struct)})")
    else:
        target_struct, target_resolved_name = load_target_from_eterna100(
            target_name=target_name,
            source=target_source,
            fallback_target=DEFAULT_TARGET_STRUCT,
            verbose=bool(cfg.get("verbose", True)),
        )

    dim = len(target_struct)
    budget = int(cfg.get("budget", 10000))
    seed = int(cfg.seed)
    nb_restarts = int(cfg.get("nb_restarts", 10))
    step_record = int(cfg.get("step_record", 100))
    step_record = max(1, step_record)
    algo_name = str(cfg.get("nevergrad_algo", "DiscreteDE"))
    show_progress = bool(cfg.get("verbose", True))

    print(
        f"Nevergrad config: algo={algo_name} target={target_resolved_name} len={dim} "
        f"budget={budget} nb_restarts={nb_restarts} step_record={step_record}"
    )

    config_name = f"nevergrad__algo{_slugify(algo_name)}__budget{_slugify(budget)}__restarts{_slugify(nb_restarts)}"
    out_dir = os.path.join(repo_root, "results", "config", config_name, "viennarna", _slugify(target_resolved_name))
    print(f"Output dir: {out_dir}")

    all_final_scores = []
    all_checkpoints = []
    with tqdm(
        total=nb_restarts * budget,
        desc=f"{algo_name} evals",
        unit="eval",
        disable=not show_progress,
    ) as pbar:
        for restart in range(nb_restarts):
            restart_seed = seed + restart
            final_score, checkpoints = _run_restart(
                target_struct=target_struct,
                dim=dim,
                budget=budget,
                step_record=step_record,
                algo_name=algo_name,
                seed=restart_seed,
                progress_bar=pbar,
            )
            all_final_scores.append(final_score)
            all_checkpoints.append(checkpoints)
            print(f"[Restart {restart + 1}/{nb_restarts}] final_best_score={final_score}")

    num_rows = max(len(row) for row in all_checkpoints)
    table = np.full((num_rows, nb_restarts), np.nan, dtype=np.float32)
    for col, row in enumerate(all_checkpoints):
        table[: len(row), col] = np.asarray(row, dtype=np.float32)
    for col in range(nb_restarts):
        col_vals = table[:, col]
        last = np.nan
        for i in range(num_rows):
            if np.isnan(col_vals[i]):
                col_vals[i] = last
            else:
                last = col_vals[i]
        if np.isnan(col_vals[0]):
            col_vals[:] = -float("inf")
        table[:, col] = col_vals

    runtime = [(i + 1) * step_record for i in range(num_rows)]
    if num_rows > 0:
        runtime[-1] = budget
    score_mean = np.mean(table, axis=1)
    score_median = np.percentile(table, 50, axis=1)
    score_std = np.std(table, axis=1)
    score_p2 = np.percentile(table, 2, axis=1)
    score_p5 = np.percentile(table, 5, axis=1)
    score_p10 = np.percentile(table, 10, axis=1)
    score_p25 = np.percentile(table, 25, axis=1)
    score_p75 = np.percentile(table, 75, axis=1)
    score_p90 = np.percentile(table, 90, axis=1)
    score_p95 = np.percentile(table, 95, axis=1)
    score_p98 = np.percentile(table, 98, axis=1)

    history = dict(
        runtime=runtime,
        best_fitness=score_mean.tolist(),
        avg_hamming=[0.0] * num_rows,
        avg_l1=[0.0] * num_rows,
        avg_entropy=[0.0] * num_rows,
        score_mean=score_mean.tolist(),
        score_median=score_median.tolist(),
        score_std=score_std.tolist(),
        score_p2=score_p2.tolist(),
        score_p5=score_p5.tolist(),
        score_p10=score_p10.tolist(),
        score_p25=score_p25.tolist(),
        score_p50=score_median.tolist(),
        score_p75=score_p75.tolist(),
        score_p90=score_p90.tolist(),
        score_p95=score_p95.tolist(),
        score_p98=score_p98.tolist(),
    )

    scores_array = np.asarray(all_final_scores, dtype=np.float32)
    print(f"average_test_score: {float(np.mean(scores_array))}")
    print("Objective: score = -ensemble_defect(target), higher is better (best possible: 0.0).")

    _save_history_csv(out_dir, history)
    _save_raw_scores_csv(out_dir, scores_array)


if __name__ == "__main__":
    main()
