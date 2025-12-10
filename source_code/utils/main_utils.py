import csv
import datetime
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def flat_or_matrix_to_instances(list_scores, nb_instances, nb_restarts):
    import numpy as _np

    def _is_num(x):
        return isinstance(x, (int, float, _np.number))

    def _as_array(x):
        try:
            import torch as _torch
            if isinstance(x, _torch.Tensor):
                x = x.detach().cpu().numpy()
        except Exception:
            pass
        if isinstance(x, (list, tuple)):
            try:
                x = _np.array(x, dtype=float)
            except Exception:
                return x
        return x

    candidate = list_scores
    if isinstance(candidate, tuple) and len(candidate) >= 1:
        if len(candidate) >= 2:
            maybe = _as_array(candidate[1])
            candidate = maybe if isinstance(maybe, _np.ndarray) else _as_array(candidate[0])
        else:
            candidate = _as_array(candidate[0])
    elif isinstance(candidate, dict):
        for k in ("scores", "list_scores", "values", "arr", "data"):
            if k in candidate:
                candidate = _as_array(candidate[k])
                break

    arr = _as_array(candidate)

    if isinstance(arr, _np.ndarray):
        if arr.ndim == 1:
            L = arr.shape[0]
            if L == nb_instances * nb_restarts:
                return [arr[i * nb_restarts:(i + 1) * nb_restarts].tolist() for i in range(nb_instances)]
            if L == nb_instances and nb_restarts == 1:
                return [[float(v)] for v in arr.tolist()]
        elif arr.ndim == 2:
            h, w = arr.shape
            if h == nb_instances and w == nb_restarts:
                return arr.tolist()
            if h == nb_restarts and w == nb_instances:
                return arr.T.tolist()
        if arr.size == nb_instances * nb_restarts:
            return arr.reshape(nb_instances, nb_restarts).tolist()
        raise ValueError(f"Format list_scores non supporté (numpy): shape={arr.shape}")

    if isinstance(candidate, (list, tuple)):
        if candidate and isinstance(candidate[0], (list, tuple)):
            if len(candidate) == nb_instances and all(len(row) == nb_restarts for row in candidate):
                return [[float(x) for x in row] for row in candidate]
            if len(candidate) == nb_restarts and all(len(col) == nb_instances for col in candidate):
                transposed = list(map(list, zip(*candidate)))
                return [[float(x) for x in row] for row in transposed]
        if all(_is_num(x) for x in candidate):
            L = len(candidate)
            if L == nb_instances * nb_restarts:
                out, idx = [], 0
                for _ in range(nb_instances):
                    out.append([float(candidate[idx + j]) for j in range(nb_restarts)])
                    idx += nb_restarts
                return out
            if L == nb_instances and nb_restarts == 1:
                return [[float(v)] for v in candidate]

    t = type(list_scores).__name__
    head = None
    try:
        head = str(candidate[:2])
    except Exception:
        head = "<nd>"
    raise ValueError(f"Format list_scores non supporté (type={t}). Aperçu={head}")


def load_grid_settings(path: str):
    """
    Charge un fichier JSON ou YAML avec des listes pour agent_M / agent_lambda.
    Retourne un dict; ignore silencieusement les clés absentes.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"grid settings file not found: {path}")

    try:
        if path.lower().endswith((".yml", ".yaml")):
            try:
                import yaml  # type: ignore
            except Exception as exc:
                raise RuntimeError("PyYAML required for YAML grid settings") from exc
            data = yaml.safe_load(Path(path).read_text())
        else:
            data = json.loads(Path(path).read_text())
    except Exception as exc:
        raise RuntimeError(f"Failed to parse grid settings file: {path}") from exc

    allowed = {}
    for key in ("agent_M", "agent_lambda"):
        if key in data:
            if not isinstance(data[key], (list, tuple)):
                raise ValueError(f"{key} in {path} must be a list/tuple")
            allowed[key] = data[key]
    return allowed


def rank_vs_global_ranking(repo_root: str, dim: int, type_instance: int, my_score: float):
    path = os.path.join(
        repo_root, "additional_results", "global_ranking",
        f"UBQP_N_{dim}_K_{type_instance}_ranks.csv"
    )
    if not os.path.isfile(path):
        return None, None, None, 0, None

    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = [row for row in reader if row]
        if not rows:
            return None, None, None, 0, None

        header = [h.strip().lower() for h in rows[0]]
        rows = rows[1:]

        algo_candidates = ["name_algo", "algo", "algorithm", "name"]
        score_candidates = ["score", "best_score", "value", "objective", "obj"]

        def find_idx(cands):
            for c in cands:
                if c in header:
                    return header.index(c)
            return None

        idx_algo = find_idx(algo_candidates)
        idx_score = find_idx(score_candidates)
        if idx_score is None:
            return None, None, None, 0, None

        entries = []
        for r in rows:
            if idx_score >= len(r):
                continue
            try:
                s = float(r[idx_score])
            except Exception:
                continue
            name = r[idx_algo] if (idx_algo is not None and idx_algo < len(r)) else "unknown"
            entries.append((name, s))

        if not entries:
            return None, None, None, 0, None

        scores_only = [s for _, s in entries]
        n = len(scores_only)

        frac_pos = sum(1 for s in scores_only if s > 0) / max(1, n)
        flip_sign = (frac_pos > 0.8 and my_score < 0)

        def to_cmp(v):
            return (-v) if flip_sign else v

        best_algo, best_score = max(entries, key=lambda t: t[1])

        my_cmp = to_cmp(my_score)
        count_gt = sum(1 for s in scores_only if s > my_cmp)
        my_rank = 1 + count_gt
        my_rank = min(max(1, my_rank), n)
        my_percentile = 100.0 * (n - my_rank + 1) / n if n > 0 else None

        return best_algo, best_score, my_rank, n, my_percentile

    except Exception:
        return None, None, None, 0, None


def load_existing_overview(path: str):
    """
    Lit best_algo_overview.csv si présent et renvoie
    {(problem, dim, type_instance): {...}}.
    """
    existing = {}
    if not os.path.isfile(path):
        return existing

    try:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            rows = [row for row in reader if row]
    except Exception:
        return existing

    if len(rows) <= 1:
        return existing

    header = [col.strip().lower() for col in rows[0]]
    try:
        idx_problem = header.index("problem")
        idx_dim = header.index("dim")
        idx_type = header.index("type_instance")
    except ValueError:
        return existing

    idx_algo = None
    for kw in ("winner_algo_key", "winner_algo"):
        if kw in header:
            idx_algo = header.index(kw)
            break
    if idx_algo is None:
        return existing

    try:
        idx_score = header.index("winner_avg_score")
    except ValueError:
        return existing

    for row in rows[1:]:
        if len(row) <= idx_score:
            continue
        problem = row[idx_problem].strip()
        try:
            dim = int(row[idx_dim])
            type_instance = int(row[idx_type])
            score = float(row[idx_score])
        except Exception:
            continue
        winner_algo = row[idx_algo].strip()
        existing[(problem, dim, type_instance)] = dict(
            problem=problem,
            dim=dim,
            type_instance=type_instance,
            winner_algo=winner_algo,
            winner_avg_score=score,
        )
    return existing


def extract_best_fitness(values):
    """
    Retourne la meilleure (min) valeur numérique dans une séquence.
    """
    if not values:
        return None

    best_val = None
    for val in values:
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if np.isnan(v):
            continue
        best_val = v if best_val is None else min(best_val, v)
    return best_val


def existing_history_best(csv_path: Path):
    """
    Lit un fichier d'historique et renvoie la dernière valeur valide de
    `best_fitness` (monotone décroissante => dernière entrée = best).
    """
    if not csv_path.exists():
        return None
    try:
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            rows = [row for row in reader if row]
    except Exception:
        return None
    if len(rows) <= 1:
        return None

    header = [col.strip().lower() for col in rows[0]]
    try:
        idx_best = header.index("best_fitness")
    except ValueError:
        return None

    for row in reversed(rows[1:]):
        if idx_best >= len(row):
            continue
        try:
            val = float(row[idx_best])
        except (TypeError, ValueError):
            continue
        if np.isnan(val):
            continue
        return val
    return None


def format_algo_display(algo_key: str) -> str:
    """
    Rend le nom d'algo lisible pour overview/summary.
    Supprime Beta/BetaAdapt/lambdaPerAgent et affiche lr_svgd.
    """
    if not algo_key:
        return "unknown"
    parts = [p for p in algo_key.split(":") if p]
    if not parts:
        return "unknown"

    update = parts[0]
    strategy = parts[1] if len(parts) > 1 else ""
    display_bits = []
    if strategy:
        display_bits.append(f"{update} @ {strategy}")
    else:
        display_bits.append(update)

    def add(label: str, value: str):
        display_bits.append(f"{label}={value}" if value != "" else label)

    for token in parts[2:]:
        clean = token.strip()
        lower = clean.lower()
        if not clean:
            continue
        if lower.startswith("betaadapt") or lower.startswith("beta"):
            continue
        if lower.startswith("lambdaperagent"):
            continue
        if lower.startswith("lambdatotal"):
            add("lambda", clean[len("lambdaTotal"):])
            continue
        if lower.startswith("lambda"):
            add("lambda", clean[len("lambda"):])
            continue
        if lower.startswith("lr_svgd"):
            add("lr_svgd", clean[len("lr_svgd"):])
            continue
        if lower.startswith("lr"):
            add("lr", clean[len("lr"):])
            continue
        if lower.startswith("delta"):
            add("delta", clean[len("delta"):])
            continue
        if lower.startswith("k"):
            add("K", clean[1:])
            continue
        if lower.startswith("m"):
            add("M", clean[1:])
            continue
        add(clean, "")

    return " | ".join(display_bits)


def write_realtime_aggregation(
    repo_root: str,
    agg_outdir: str,
    per_inst_algo_avg: dict,
    run_histories: dict | None = None,
):
    """
    Écrit/Met à jour en temps réel:
      - best_algo_overview.csv (par groupe: winner + rank/percent + winner_avg_score)
      - best_algo_summary.txt  (lisible humain)
    """
    n_instances_seen = defaultdict(int)
    for (problem, dim, type_instance, inst_idx) in per_inst_algo_avg.keys():
        n_instances_seen[(problem, dim, type_instance)] = max(
            n_instances_seen[(problem, dim, type_instance)],
            inst_idx + 1
        )

    groups = sorted({(p, d, t) for (p, d, t, _) in per_inst_algo_avg.keys()},
                    key=lambda x: (x[0], x[1], x[2]))

    overview_csv = os.path.join(agg_outdir, "best_algo_overview.csv")
    existing_entries = load_existing_overview(overview_csv)
    new_entries = {}
    history_by_group = {}

    for (problem, dim, type_instance) in groups:
        n_inst = n_instances_seen.get((problem, dim, type_instance), 0)
        algo_to_avgs = defaultdict(list)
        for inst_idx in range(n_inst):
            key_inst = (problem, dim, type_instance, inst_idx)
            for algo_key, avg_val in per_inst_algo_avg.get(key_inst, {}).items():
                if not (avg_val is None or np.isnan(avg_val)):
                    algo_to_avgs[algo_key].append(avg_val)

        if not algo_to_avgs:
            continue

        algo_mean = {
            algo: float(np.mean(vals)) for algo, vals in algo_to_avgs.items() if len(vals) > 0
        }
        if not algo_mean:
            continue

        winner_algo, winner_avg_score = min(algo_mean.items(), key=lambda kv: kv[1])
        group_key = (problem, int(dim), int(type_instance))
        new_entries[group_key] = dict(
            problem=problem,
            dim=int(dim),
            type_instance=int(type_instance),
            winner_algo=winner_algo,
            winner_avg_score=float(winner_avg_score),
        )
        if run_histories:
            history = run_histories.get((problem, dim, type_instance, winner_algo))
            if history:
                history_by_group[group_key] = history

    final_entries = dict(existing_entries)
    improved_groups = set()
    for group_key, data in new_entries.items():
        new_score = data["winner_avg_score"]
        prev = existing_entries.get(group_key)
        better = False
        if prev is None:
            better = True
        else:
            prev_score = prev.get("winner_avg_score")
            if prev_score is None or np.isnan(prev_score):
                better = True
            elif np.isnan(new_score):
                better = False
            else:
                better = (new_score + 1e-12) < prev_score
        if better:
            final_entries[group_key] = {
                "problem": data["problem"],
                "dim": data["dim"],
                "type_instance": data["type_instance"],
                "winner_algo": data["winner_algo"],
                "winner_avg_score": new_score,
            }
            improved_groups.add(group_key)

    if not final_entries:
        print("=========================================================FIN==========================================================================")
        print("[INFO] Aucun overview à écrire (aucun groupe évalué).")
        return

    if not improved_groups:
        print("=========================================================FIN==========================================================================")
        print("[INFO] best_algo_overview inchangé (aucune amélioration).")
        return

    lines2 = ["problem,dim,type_instance,winner_algo_key,winner_algo,rank,percent,winner_avg_score"]
    human_lines = []
    sorted_entries = sorted(
        final_entries.values(),
        key=lambda d: (d["problem"], d["dim"], d["type_instance"])
    )
    for entry in sorted_entries:
        problem = entry["problem"]
        dim = entry["dim"]
        type_instance = entry["type_instance"]
        winner_algo = entry["winner_algo"]
        winner_display = format_algo_display(winner_algo)
        winner_avg_score = entry["winner_avg_score"]

        _best_algo_csv, _best_score_csv, my_rank, n_rank, my_pct = rank_vs_global_ranking(
            repo_root, int(dim), int(type_instance), winner_avg_score
        )
        rank_val = str(my_rank) if my_rank is not None else ""
        percent_str = f"{my_pct:.1f}" if my_pct is not None else ""

        lines2.append(
            f"{problem},{dim},{type_instance},{winner_algo},{winner_display},{rank_val},{percent_str},{winner_avg_score}"
        )
        pct_disp = f"{percent_str}%" if percent_str else "n/a"
        rank_disp = f"{rank_val}/{n_rank}" if (rank_val and n_rank is not None) else "n/a"
        human_lines.append(
            f"{problem} | dim={dim} | type_instance={type_instance} -> "
            f"{winner_display} | rank={rank_disp} ({pct_disp}) | winner_avg_score={winner_avg_score:.6f}"
        )

    Path(overview_csv).write_text("\n".join(lines2), encoding="utf-8")

    pretty_txt = os.path.join(agg_outdir, "best_algo_summary.txt")
    header = [
        "=== BEST ALGO OVERVIEW (temps réel) ===",
        f"Généré le {datetime.datetime.now().isoformat(timespec='seconds')}",
        "",
        *human_lines,
        "",
        f"CSV : {overview_csv}",
        ""
    ]
    Path(pretty_txt).write_text("\n".join(header), encoding="utf-8")

    if improved_groups:
        for group_key in improved_groups:
            history = history_by_group.get(group_key)
            if not history:
                continue
            entry = final_entries[group_key]
            _write_group_history(
                agg_outdir,
                entry["problem"],
                entry["dim"],
                entry["type_instance"],
                entry["winner_algo"],
                history,
            )

    print("=========================================================FIN==========================================================================\n")


def _write_group_history(agg_outdir: str, problem: str, dim: int, type_instance: int, algo_key: str, history: dict):
    if not history:
        return

    runtimes = history.get("runtime") or []
    best_fitness = history.get("best_fitness") or []
    avg_hamming = history.get("avg_hamming") or []
    avg_js = history.get("avg_js") or []
    avg_l2 = history.get("avg_l2") or []

    length = max(len(runtimes), len(best_fitness), len(avg_hamming), len(avg_js), len(avg_l2))
    if length == 0:
        return

    new_best = extract_best_fitness(best_fitness)

    def _safe(arr, i):
        return arr[i] if i < len(arr) else ""

    safe_algo = algo_key.replace(":", "_").replace("/", "_")
    out_dir = Path(agg_outdir) / "instance_history"
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{problem}_{dim}_{type_instance}_"
    existing_files = list(out_dir.glob(f"{prefix}*.csv"))
    existing_best = None
    for f in existing_files:
        existing_best = existing_history_best(f)
        if existing_best is not None:
            break

    if new_best is not None and existing_best is not None:
        if not (new_best + 1e-12 < existing_best):
            return
    elif new_best is None and existing_best is not None:
        return

    for f in existing_files:
        try:
            f.unlink()
        except OSError:
            pass

    out_file = out_dir / f"{prefix}{safe_algo}.csv"
    lines = ["runtime,best_fitness,avg_hamming,avg_js,avg_l2"]
    for i in range(length):
        lines.append(
            f"{_safe(runtimes, i)},{_safe(best_fitness, i)},{_safe(avg_hamming, i)},{_safe(avg_js, i)},{_safe(avg_l2, i)}"
        )
    out_file.write_text("\n".join(lines), encoding="utf-8")
