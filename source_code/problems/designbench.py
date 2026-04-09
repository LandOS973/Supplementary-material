from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf


DEFAULT_TASK_NAME = "GFP-v0"
DEFAULT_DIM = 237
DEFAULT_ALPHABET_SIZE = 20
DEFAULT_ORACLE_BATCH_SIZE = 2048


def resolve_designbench_problem(cfg) -> dict:
    problem_name = str(OmegaConf.select(cfg, "problem.name") or "").lower()
    if problem_name in ("designbench", "gfp", "gfp-v0"):
        return dict(
            used_fallback=False,
            task_name=str(OmegaConf.select(cfg, "problem.task_name") or DEFAULT_TASK_NAME),
            dim=int(OmegaConf.select(cfg, "problem.dim") or DEFAULT_DIM),
            alphabet_size=int(OmegaConf.select(cfg, "problem.alphabet_size") or DEFAULT_ALPHABET_SIZE),
            oracle_batch_size=int(
                OmegaConf.select(cfg, "problem.oracle_batch_size")
                or cfg.get("oracle_batch_size")
                or DEFAULT_ORACLE_BATCH_SIZE
            ),
        )
    return dict(
        used_fallback=True,
        task_name=DEFAULT_TASK_NAME,
        dim=DEFAULT_DIM,
        alphabet_size=DEFAULT_ALPHABET_SIZE,
        oracle_batch_size=int(cfg.get("oracle_batch_size", DEFAULT_ORACLE_BATCH_SIZE)),
    )


def load_designbench_task(task_name: str, use_cuda: bool = False):
    try:
        import design_bench
    except Exception as exc:
        raise RuntimeError("DesignBench requires `design-bench` installed in the active environment.") from exc
    try:
        return design_bench.make(task_name, use_cuda=use_cuda)
    except TypeError:
        return design_bench.make(task_name)


def infer_task_space(task_obj, dim_val: int, alpha_val: int):
    x_attr = getattr(task_obj, "x", None)
    if x_attr is None:
        return dim_val, alpha_val, None, True
    try:
        x_np = np.asarray(x_attr)
    except Exception:
        return dim_val, alpha_val, None, True
    if x_np.ndim == 3:
        return int(x_np.shape[1]), int(x_np.shape[2]), "onehot", True
    if x_np.ndim == 2:
        inferred_dim = int(x_np.shape[1])
        if np.issubdtype(x_np.dtype, np.integer):
            if np.min(x_np) < 0:
                return inferred_dim, alpha_val, "tokens", False
            return inferred_dim, int(np.max(x_np)) + 1, "tokens", True
        rounded = np.rint(x_np)
        max_abs = float(np.max(np.abs(x_np - rounded)))
        if max_abs < 1e-6 and np.min(rounded) >= 0:
            return inferred_dim, int(np.max(rounded)) + 1, "tokens", True
        return inferred_dim, alpha_val, "tokens", False
    return dim_val, alpha_val, None, False


def oracle_sanity_check(task, dim: int, alphabet_size: int, verbose: bool = True) -> None:
    if not verbose:
        return
    oracle_fn = getattr(task, "predict", None)
    if not callable(oracle_fn):
        oracle_fn = getattr(task, "score", None)
    if not callable(oracle_fn):
        print("[OracleCheck] No callable oracle found on task.")
        return

    def _fmt_stats(label: str, arr: np.ndarray) -> None:
        arr = np.asarray(arr, dtype=np.float32)
        print(
            f"{label} mean={float(arr.mean()):.6g} std={float(arr.std()):.6g} "
            f"min={float(arr.min()):.6g} max={float(arr.max()):.6g}"
        )

    try:
        rng = np.random.default_rng()
        sample_n = 128
        tokens = rng.integers(0, alphabet_size, size=(sample_n, dim), endpoint=False, dtype=np.int64)
        mode = "tokens"
        try:
            scores = np.asarray(oracle_fn(tokens), dtype=np.float32)
        except Exception:
            onehot = np.eye(alphabet_size, dtype=np.float32)[tokens]
            scores = np.asarray(oracle_fn(onehot), dtype=np.float32)
            mode = "onehot"
        print(f"[OracleCheck] random_input mode={mode} shape={tuple(scores.shape)}")
        _fmt_stats("[OracleCheck] random_input", scores)
    except Exception as exc:
        print(f"[OracleCheck] failed: {exc}")

    try:
        x_attr = getattr(task, "x", None)
        if x_attr is not None:
            try:
                x_sample = x_attr[:128]
            except Exception:
                x_sample = x_attr
            x_np = np.asarray(x_sample)
            x_mode = "unknown"
            if x_np.ndim == 3 and x_np.shape[-1] == alphabet_size:
                x_mode = "onehot"
            elif x_np.ndim == 2:
                x_mode = "tokens"
            if x_mode != "unknown":
                try:
                    scores_x = np.asarray(oracle_fn(x_np), dtype=np.float32)
                    print(f"[OracleCheck] task.x mode={x_mode} shape={tuple(scores_x.shape)}")
                    _fmt_stats("[OracleCheck] task.x", scores_x)
                except Exception as exc:
                    print(f"[OracleCheck] task.x predict failed: {exc}")
            else:
                print(f"[OracleCheck] task.x shape={tuple(x_np.shape)} (unsupported format)")
    except Exception as exc:
        print(f"[OracleCheck] task.x probe failed: {exc}")

    try:
        y_attr = getattr(task, "y", None)
        if y_attr is not None:
            try:
                y_sample = y_attr[:128]
            except Exception:
                y_sample = y_attr
            y_np = np.asarray(y_sample, dtype=np.float32)
            print(f"[OracleCheck] task.y shape={tuple(y_np.shape)}")
            _fmt_stats("[OracleCheck] task.y", y_np)
    except Exception as exc:
        print(f"[OracleCheck] task.y probe failed: {exc}")
