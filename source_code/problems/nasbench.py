from __future__ import annotations

import os
from pathlib import Path


NASBENCH_DIM = 26
NASBENCH_DIM_VARIABLES = [2 for _ in range(21)] + [3 for _ in range(5)]


def is_nasbench_problem(problem_name: str) -> bool:
    return str(problem_name).strip().lower() in ("nasbench", "nasbench_full")


def resolve_nasbench_data_file(script_dir: str | Path, data_file: str | None) -> str:
    script_dir = str(script_dir)
    if data_file:
        if os.path.isabs(data_file):
            return data_file
        return os.path.join(script_dir, data_file)
    candidate = os.path.join(script_dir, "instances", "nasbench", "nasbench_full.tfrecord")
    if os.path.exists(candidate):
        return candidate
    return "nasbench_full.tfrecord"


def load_nasbench_objective(data_file: str):
    try:
        from bbdob import NasBench101
    except Exception as exc:
        raise RuntimeError(
            "NasBench requires bbdob. Install it with `pip install -e .` in the BB-DOB repo."
        ) from exc
    return NasBench101(filename=data_file)
