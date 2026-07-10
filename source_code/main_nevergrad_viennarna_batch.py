"""
Run ONE Nevergrad discrete optimizer on ONE ViennaRNA (Eterna100) target,
`nb_restarts` times, and store one per-run curve file per restart.

Output layout mirrors the NK/QUBO Nevergrad results, but with a single-level
problem path (VIENNARNA/<target>):

    results/nevergrad/<algo>/VIENNARNA/<target_slug>/
        results_nevergrad_<algo>_VIENNARNA_<target_slug>_budget_<budget>_<ts>_i_0_r_<r>.txt

Each file has the same 2-column format as the NK runs:

    runtime, score
    100,<best_so_far>
    200,<best_so_far>
    ...

Objective: score = -ensemble_defect(target)  (maximised, best possible = 0.0),
identical to main_nevergrad_viennarna.py so curves are directly comparable.

Example:
    python main_nevergrad_viennarna_batch.py DiscreteDE
    python main_nevergrad_viennarna_batch.py DiscreteDE --target "Simple Hairpin" --nb_restarts 10
"""

from __future__ import annotations

import argparse
import datetime
import os
import random
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from main_nevergrad_viennarna import (
    DEFAULT_TARGET_STRUCT,
    _score_tokens,
    _slugify,
)
from problems.viennarna import (
    ETERNA100_TSV_URL,
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


def _resolve_registry_key(name: str) -> str:
    """Return the exact registry key for `name`, tolerating the trailing-space
    quirk of some nevergrad keys (e.g. 'UltraSmoothDiscreteLognormalOnePlusOne ')."""
    reg = ng.optimizers.registry
    for key in (name, name + " "):
        if reg.get(key) is not None:
            return key
    available = sorted(k.strip() for k in reg.keys())
    raise ValueError(f"Unknown Nevergrad algo '{name}'. Available: {available}")


def _run_restart_streaming(
    target_struct: str,
    dim: int,
    budget: int,
    step_record: int,
    registry_key: str,
    seed: int,
    out_path: str,
) -> float:
    """Run one restart, writing each checkpoint to `out_path` as it is reached
    (flushed immediately). If the job is killed mid-restart, the file keeps all
    checkpoints produced so far. Returns the final best score."""
    param = ng.p.TransitionChoice(range(4), repetitions=dim, ordered=False)
    algo_cls = ng.optimizers.registry.get(registry_key)
    optimizer = algo_cls(parametrization=param, budget=budget)
    optimizer.parametrization.random_state.seed(seed)

    best_score = -float("inf")
    with open(out_path, "w") as f:
        f.write("runtime, score\n")
        for step in range(1, budget + 1):
            candidate = optimizer.ask()
            score = _score_tokens(np.asarray(candidate.value), target_struct)
            optimizer.tell(candidate, -score)
            if score > best_score:
                best_score = score
            if step % step_record == 0:
                f.write(f"{step},{best_score}\n")
                f.flush()
        if budget % step_record != 0:
            f.write(f"{budget},{best_score}\n")
            f.flush()
    return best_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Nevergrad discrete algo on a ViennaRNA target")
    parser.add_argument("name_algo", type=str, help="Nevergrad optimizer name (e.g. DiscreteDE)")
    parser.add_argument("--target", type=str, default="Simple Hairpin",
                        help="Eterna100 target name (default: Simple Hairpin)")
    parser.add_argument("--target_source", type=str, default=None,
                        help="TSV url or local path for the Eterna100 benchmark "
                             "(default: local cached TSV in problems/, else the GitHub URL)")
    parser.add_argument("--target_struct", type=str, default=None,
                        help="Explicit dot-bracket target; overrides --target if set")
    parser.add_argument("--nb_restarts", type=int, default=10, help="number of independent runs")
    parser.add_argument("--seed", type=int, default=0, help="base random seed (restart r uses seed+r)")
    parser.add_argument("--budget", type=int, default=50000, help="function evaluations per run")
    parser.add_argument("--step_record", type=int, default=100, help="record best-so-far every N evals")
    args = parser.parse_args()

    if RNA is None:
        raise RuntimeError("Import `RNA` failed. Install ViennaRNA bindings (`pip install ViennaRNA`).")
    if ng is None:
        raise RuntimeError("Import `nevergrad` failed. Install Nevergrad (`pip install nevergrad`).")

    name_algo = args.name_algo.strip()
    registry_key = _resolve_registry_key(name_algo)  # fail fast on a bad name

    # ── resolve target structure ────────────────────────────────────────────
    if args.target_struct:
        target_struct = normalize_target_struct(args.target_struct)
        target_name = "cfg_target_struct"
    else:
        # Prefer the TSV cached in the repo: Jean Zay compute nodes have NO
        # internet, so fetching from the URL would fail and silently fall back
        # to a wrong default target.
        local_tsv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "problems", "eterna100_puzzles.tsv")
        target_source = args.target_source or (local_tsv if os.path.exists(local_tsv) else ETERNA100_TSV_URL)
        target_struct, target_name = load_target_from_eterna100(
            target_name=args.target,
            source=target_source,
            fallback_target=DEFAULT_TARGET_STRUCT,
            verbose=True,
        )
        # Never run on the silent fallback target: fail loudly instead of
        # producing results for the wrong structure.
        if target_name == "fallback_default":
            raise RuntimeError(
                f"Could not load target '{args.target}' from '{target_source}'. "
                "Refusing to run on the fallback default target. On Jean Zay ensure "
                "the cached TSV 'source_code/problems/eterna100_puzzles.tsv' exists "
                "(compute nodes have no internet)."
            )

    dim = len(target_struct)
    budget = int(args.budget)
    step_record = max(1, int(args.step_record))
    nb_restarts = int(args.nb_restarts)
    seed = int(args.seed)
    target_slug = _slugify(target_name)

    np.random.seed(seed)
    random.seed(seed)

    script_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    out_dir = os.path.join(repo_root, "results", "nevergrad", name_algo, "VIENNARNA", target_slug)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(f"[nevergrad-viennarna] algo={name_algo} target={target_name} len={dim} "
          f"budget={budget} step_record={step_record} nb_restarts={nb_restarts}")
    print(f"Output dir: {out_dir}")

    for r in range(nb_restarts):
        restart_seed = seed + r
        filename = (
            f"results_nevergrad_{name_algo}_VIENNARNA_{target_slug}_budget_{budget}"
            f"_{timestamp}_i_0_r_{r}.txt"
        )
        out_path = os.path.join(out_dir, filename)
        best_score = _run_restart_streaming(
            target_struct=target_struct,
            dim=dim,
            budget=budget,
            step_record=step_record,
            registry_key=registry_key,
            seed=restart_seed,
            out_path=out_path,
        )
        print(f"  [r={r}] final best score={best_score:.6f}")


if __name__ == "__main__":
    main()
