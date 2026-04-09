"""
Rebuild overall_summary.xlsx by scanning results/config/* folders.
"""

import csv
import os
import re
from pathlib import Path

import numpy as np
from openpyxl import Workbook


SUMMARY_HEADERS = [
    "config_name",
    "kernel",
    "advantage",
    "M",
    "lambda_",
    "epsilon_svgd",
    "gamma",
    "decay_start_ratio",
    "decay_min_factor",
    "nasbench_avg_score",
    "mean_rank",
    "median_rank",
    "std_percent",
    "top1_count",
    "top3_count",
    "top5_count",
    "top10_count",
    "top_1_nk",
    "top_1_nk3",
    "top_1_qubo",
    "win_rate_mean",
    "mean_hamming_norm",
    "mean_l1_norm",
    "hasRawScore",
    "n_instances",
    "n_ranked",
]


INSTANCE_DIR_RE = re.compile(r"^(?P<problem>QUBO|NK|NK3)_dim(?P<dim>\d+)_t(?P<t>\d+)$")


def _round_float(value, digits: int = 6):
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except Exception:
        return None


def _infer_params_from_name(config_name: str) -> dict:
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
        "bandwith_kernel": None,
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
        elif p.startswith("bw"):
            out["bandwith_kernel"] = parse_float(p[2:])
    return out


def _has_raw_score(cfg_dir: Path) -> int:
    return 1 if any(cfg_dir.rglob("raw_scores.csv")) else 0


def _nasbench_avg_score(cfg_dir: Path):
    metrics_path = cfg_dir / "nasbench" / "best_metrics.csv"
    if not metrics_path.is_file():
        return None
    try:
        with metrics_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader if row]
        if not rows:
            return None
        last = rows[-1]
        for col in ("mean", "median", "best_fitness"):
            if col in last and last[col] not in (None, ""):
                return float(last[col])
    except Exception:
        return None


def _global_ranking_path(repo_root: Path, problem: str, dim: int, t: int) -> Path:
    if problem == "QUBO":
        return repo_root / "additional_results" / "global_ranking" / f"UBQP_N_{dim}_K_{t}_ranks.csv"
    return repo_root / "additional_results" / "global_ranking" / f"{problem}_N_{dim}_K_{t}_ranks.csv"


def _load_global_ranking_scores(path: Path, exclude_algo: str = "PPO-EDA") -> list[tuple[str, float]]:
    if not path.is_file():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                if not row:
                    continue
                name = (row.get("name_algo") or row.get("algo") or row.get("algorithm") or row.get("name") or "").strip()
                if not name or name.lower() == exclude_algo.lower():
                    continue
                raw = row.get("score") or row.get("best_score") or row.get("value") or row.get("objective") or row.get("obj")
                try:
                    score = float(raw)
                except Exception:
                    continue
                rows.append((name, score))
            return rows
    except Exception:
        return []


def _read_last_metrics_row(metrics_path: Path) -> dict[str, str] | None:
    if not metrics_path.is_file():
        return None
    try:
        with metrics_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader if row]
        if not rows:
            return None
        return rows[-1]
    except Exception:
        return None


def _pick_float(row: dict[str, str] | None, cols: tuple[str, ...]) -> float | None:
    if not row:
        return None
    for col in cols:
        val = row.get(col)
        if val in (None, ""):
            continue
        try:
            return float(val)
        except Exception:
            continue
    return None


def _expected_instances(config_path: Path, ranking_dir: Path) -> list[tuple[str, int, int]]:
    expected = []
    for rank_file in sorted(ranking_dir.glob("*_ranks.csv")):
        match = re.match(r"^(?P<problem>UBQP|NK|NK3)_N_(?P<dim>\d+)_K_(?P<t>\d+)_ranks\.csv$", rank_file.name)
        if not match:
            continue
        problem = match.group("problem")
        if problem == "UBQP":
            problem = "QUBO"
        expected.append((problem, int(match.group("dim")), int(match.group("t"))))
    if expected:
        return sorted(expected, key=lambda item: (item[0], item[1], item[2]))

    for child in sorted(config_path.iterdir()):
        if not child.is_dir():
            continue
        match = INSTANCE_DIR_RE.match(child.name)
        if not match:
            continue
        expected.append((match.group("problem"), int(match.group("dim")), int(match.group("t"))))
    return sorted(expected, key=lambda item: (item[0], item[1], item[2]))


def _collect_config_stats_from_csv(config_dir: Path, config_name: str, params: dict, repo_root: Path) -> dict:
    rows = []
    ranking_dir = repo_root / "additional_results" / "global_ranking"
    expected_instances = _expected_instances(config_dir, ranking_dir)

    for problem, dim, t in expected_instances:
        instance_dir = config_dir / f"{problem}_dim{dim}_t{t}"
        metrics_path = instance_dir / "best_metrics.csv"
        if not metrics_path.is_file():
            legacy_metrics = instance_dir / f"{problem}_{params['kernel']}_best_metrics.csv"
            metrics_path = legacy_metrics if legacy_metrics.is_file() else None
        if metrics_path is None:
            continue

        metrics_last = _read_last_metrics_row(metrics_path)
        avg_score = _pick_float(metrics_last, ("mean", "median", "best_fitness"))
        if avg_score is None:
            continue

        ranking_path = _global_ranking_path(repo_root, problem, dim, t)
        ranking_entries = _load_global_ranking_scores(ranking_path)
        if ranking_entries:
            scores_only = [score for _, score in ranking_entries]
            frac_pos = sum(1 for s in scores_only if s > 0) / max(1, len(scores_only))
            flip_sign = frac_pos > 0.8 and avg_score < 0
            my_cmp = -avg_score if flip_sign else avg_score
            best_algo, best_score = max(ranking_entries, key=lambda x: x[1])
            n_rank = len(scores_only)
            my_rank = 1 + sum(1 for s in scores_only if s > my_cmp)
            my_rank = min(max(1, my_rank), n_rank)
            my_pct = 100.0 * (n_rank - my_rank + 1) / n_rank if n_rank > 0 else None
        else:
            best_algo = None
            best_score = None
            n_rank = 0
            my_rank = None
            my_pct = None

        win_rate = ((n_rank - my_rank) / n_rank) if (my_rank is not None and n_rank > 0) else None
        hamming_norm = _pick_float(metrics_last, ("avg_hamming",))
        l1_norm = _pick_float(metrics_last, ("avg_l1",))
        if hamming_norm is not None and hamming_norm > 1:
            hamming_norm = hamming_norm / dim
        if l1_norm is not None and l1_norm > 1:
            l1_norm = l1_norm / dim

        rows.append(
            dict(
                problem=problem,
                dim=dim,
                type_instance=t,
                avg_score=avg_score,
                rank=my_rank,
                percent=my_pct,
                top1_count=1 if my_rank == 1 else 0,
                top3_count=1 if my_rank is not None and my_rank <= 3 else 0,
                top5_count=1 if my_rank is not None and my_rank <= 5 else 0,
                top10_count=1 if my_rank is not None and my_rank <= 10 else 0,
                ranking_best_algo=best_algo,
                ranking_best_score=best_score,
                n_rank=n_rank,
                win_rate=win_rate,
                hamming_norm=hamming_norm,
                l1_norm=l1_norm,
            )
        )

    ranks = [r["rank"] for r in rows if r.get("rank") is not None]
    percents = [r["percent"] for r in rows if r.get("percent") is not None]
    win_rates = [r["win_rate"] for r in rows if r.get("win_rate") is not None]
    hamming_vals = [r["hamming_norm"] for r in rows if r.get("hamming_norm") is not None]
    l1_vals = [r["l1_norm"] for r in rows if r.get("l1_norm") is not None]

    return dict(
        config_name=config_name,
        kernel=params["kernel"],
        advantage=params["advantage"],
        M=params["M"],
        lambda_=params["lambda_"],
        epsilon_svgd=_round_float(params["epsilon_svgd"]),
        gamma=_round_float(params["gamma"]),
        decay_start_ratio=_round_float(params["decay_start_ratio"]),
        decay_min_factor=_round_float(params["decay_min_factor"]),
        mean_rank=float(np.mean(ranks)) if ranks else None,
        median_rank=float(np.median(ranks)) if ranks else None,
        std_percent=float(np.std(percents)) if len(percents) > 1 else (0.0 if percents else None),
        top1_count=sum(1 for r in rows if r.get("top1_count")),
        top3_count=sum(1 for r in rows if r.get("top3_count")),
        top5_count=sum(1 for r in rows if r.get("top5_count")),
        top10_count=sum(1 for r in rows if r.get("top10_count")),
        top_1_nk=sum(1 for r in rows if r.get("top1_count") and r.get("problem") == "NK"),
        top_1_nk3=sum(1 for r in rows if r.get("top1_count") and r.get("problem") == "NK3"),
        top_1_qubo=sum(1 for r in rows if r.get("top1_count") and r.get("problem") == "QUBO"),
        win_rate_mean=float(np.mean(win_rates)) if win_rates else None,
        mean_hamming_norm=float(np.mean(hamming_vals)) if hamming_vals else None,
        mean_l1_norm=float(np.mean(l1_vals)) if l1_vals else None,
        n_instances=len(rows),
        n_ranked=len(ranks),
    )


def main():
    repo_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    out_root = repo_root / "results" / "config"
    out_xlsx = out_root / "overall_summary.xlsx"

    rows = []
    for cfg_dir in sorted(out_root.iterdir()):
        if not cfg_dir.is_dir():
            continue
        config_name = cfg_dir.name
        params = _infer_params_from_name(config_name)
        if not params.get("kernel") or params.get("M") is None or params.get("lambda_") is None:
            continue
        stats = _collect_config_stats_from_csv(cfg_dir, config_name, params, repo_root)
        stats["nasbench_avg_score"] = _nasbench_avg_score(cfg_dir)
        stats["hasRawScore"] = _has_raw_score(cfg_dir)
        rows.append(stats)

    wb = Workbook()
    ws = wb.active
    ws.title = "summary"
    ws.append(SUMMARY_HEADERS)
    for row in rows:
        ws.append([row.get(h) for h in SUMMARY_HEADERS])

    wb.save(str(out_xlsx))
    print(f"[DONE] Rebuilt: {out_xlsx} ({len(rows)} configs)")


if __name__ == "__main__":
    main()
