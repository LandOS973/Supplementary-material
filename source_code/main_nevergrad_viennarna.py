#!/usr/bin/env python3
"""
Hydra runner for Nevergrad DiscreteDE on ViennaRNA inverse folding.

Objective used:
score = -ensemble_defect(target)

Nevergrad minimizes by default, so we optimize loss = -score.
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from urllib.request import urlopen

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

try:
    import nevergrad as ng
except Exception:  # pragma: no cover
    ng = None

try:
    import RNA
except Exception:  # pragma: no cover
    RNA = None


RNA_ALPHABET = np.array(list("ACGU"))
ETERNA100_TSV_URL = (
    "https://raw.githubusercontent.com/eternagame/eterna100-benchmarking/master/data/eterna100_puzzles.tsv"
)
# Fallback only. Default run target should be set from Hydra config:
# `config/problem/anr.yaml` -> `target_name`.
DEFAULT_TARGET_NAME = "Simple Hairpin"
DEFAULT_TARGET_STRUCT = "(" * 25 + "." * 50 + ")" * 25


def _is_dotbracket(value: str) -> bool:
    if not value:
        return False
    allowed = set(".()[]{}<>&")
    return all(ch in allowed for ch in value)


def _normalize_target_struct(value: str) -> str:
    target = str(value).strip().replace(" ", "")
    if not target:
        raise ValueError("Target structure is empty.")
    if not _is_dotbracket(target):
        raise ValueError(f"Invalid target structure `{target}`. Expected dot-bracket symbols only.")
    return target


def _extract_dotbracket_from_row(row: dict) -> str | None:
    preferred_keys = (
        "target_structure",
        "target_struct",
        "structure",
        "secstruct",
        "secondary_structure",
        "dotbracket",
        "dot_bracket",
    )
    for key in preferred_keys:
        value = row.get(key)
        if value and _is_dotbracket(str(value).strip()):
            return str(value).strip()
    for value in row.values():
        if value and _is_dotbracket(str(value).strip()):
            return str(value).strip()
    return None


def _row_matches_name(row: dict, target_name: str) -> bool:
    ref = target_name.strip().lower()
    if not ref:
        return False
    name_keys = ("name", "puzzle_name", "title", "display_name", "id")
    for key in name_keys:
        value = row.get(key)
        if value and str(value).strip().lower() == ref:
            return True
    for value in row.values():
        if isinstance(value, str) and value.strip().lower() == ref:
            return True
    return False


def _read_text_from_source(source: str) -> str:
    source = str(source).strip()
    if source.startswith("http://") or source.startswith("https://"):
        with urlopen(source, timeout=20) as response:
            return response.read().decode("utf-8")
    return Path(source).read_text(encoding="utf-8")


def _load_target_from_eterna100(
    target_name: str = DEFAULT_TARGET_NAME,
    source: str = ETERNA100_TSV_URL,
    fallback_target: str = DEFAULT_TARGET_STRUCT,
    verbose: bool = True,
) -> tuple[str, str]:
    try:
        raw = _read_text_from_source(source)
        reader = csv.DictReader(raw.splitlines(), delimiter="\t")
        rows = list(reader)
        for row in rows:
            if _row_matches_name(row, target_name):
                candidate = _extract_dotbracket_from_row(row)
                if candidate:
                    target = _normalize_target_struct(candidate)
                    if verbose:
                        print(f"[ViennaRNA] target loaded from benchmark: `{target_name}` (len={len(target)})")
                    return target, target_name
        if verbose:
            print(
                f"[ViennaRNA][WARN] target `{target_name}` not found in benchmark source; "
                "using fallback default target."
            )
    except Exception as exc:
        if verbose:
            print(f"[ViennaRNA][WARN] failed loading benchmark target ({exc}); using fallback default target.")

    return _normalize_target_struct(fallback_target), "fallback_default"


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
        target_struct = _normalize_target_struct(str(target_struct_cfg))
        target_resolved_name = "cfg_target_struct"
        print(f"[ViennaRNA] using target from cfg.problem.target_struct (len={len(target_struct)})")
    else:
        target_struct, target_resolved_name = _load_target_from_eterna100(
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
    # forward-fill missing tail if any (only happens with uneven checkpoints)
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
