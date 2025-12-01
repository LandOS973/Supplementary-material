#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test commit depuis nautilus
import os
import csv
import datetime
import itertools
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import random
import tempfile
import argparse
import sys
import socket

# --- imports projet (identiques à ton main) ---
from eda_strategies.FactoryStrategyEA import FactoryStrategyEA
from environment.qubo import getTensorInstances_QUBO, get_Score_trajectoriesQUBO_cuda
from environment.nk import getTensorInstances_NK, get_Score_trajectoriesNK_cuda
# ------------------------------------------------

"""
Sweep avec:
- UNE SEULE barre tqdm par run (via fichier temporaire)
- Mise à jour en TEMPS RÉEL:
    - results/aggregation/best_algo_overview.csv        (winner par groupe + rank/percent + winner_avg_score)
    - results/aggregation/best_algo_summary.txt         (résumé humain)

Supprimé:
- sweep_summary.json

Conventions:
- Problème de MINIMISATION (plus bas = meilleur)
- Winner d’un groupe = algo avec la plus faible **moyenne** (winner_avg_score) sur les instances observées.
- La dernière colonne de l’overview est `winner_avg_score` (moyenne des moyennes par instance).
- On ajoute `rank`, `percent` via le fichier additional_results/global_ranking/UBQP_N_{dim}_K_{type}.csv
"""

# =========================
# 1) Defaults (comme ton config)
# =========================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

DEFAULTS = dict(
    device=device,
    seed=0,
    verbose=True,
    nb_instances_test=10,
    nb_restarts=10,
    budget=10000,
    lambda_=10,
    type_strategy="PPO-EDA",   # utilisé par la fabrique
    problem_name="QUBO",       # defaults: problem: qubo
    visualization=False,
    learning_rate_svgd=0.1,
)

# =========================
# 2) Grille d’hparams (comme hydra.sweeper.params)
# =========================
GRID = dict(
    agent=["ppo"],
    agent_learning_rate=[0.008, 0.02, 0.012, 0.03],
    agent_M=[1, 2, 4, 5],
    agent_K_steps=[6, 8, 20, 14],
    agent_delta_target=[0.0025, 0.006],
    agent_learning_rate_svgd=[0.1, 0.2, 0.5, 1.0],
    problem_dim=[64, 128, 256],
    problem_type_instance=[0, 1, 2, 3, 4, 5],
    agent_lambda=[10, 15, 20, 25],
)

# =========================
# 3) Normalisation robuste des scores par instance
# =========================
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
                return [arr[i*nb_restarts:(i+1)*nb_restarts].tolist() for i in range(nb_instances)]
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


def _load_grid_settings(path: str):
    """
    Charge un fichier JSON ou YAML avec des listes pour agent_M / agent_lambda.
    Retourne un dict; ignore silencieusement les clés absentes.
    """
    import json

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

        # Heuristique: CSV très probablement en maximisation si majoritairement positif
        frac_pos = sum(1 for s in scores_only if s > 0) / max(1, n)
        flip_sign = (frac_pos > 0.8 and my_score < 0)

        def to_cmp(v):
            return (-v) if flip_sign else v

        # Meilleur dans le CSV (max)
        best_algo, best_score = max(entries, key=lambda t: t[1])

        my_cmp = to_cmp(my_score)
        count_gt = sum(1 for s in scores_only if s > my_cmp)
        my_rank = 1 + count_gt
        my_rank = min(max(1, my_rank), n)
        my_percentile = 100.0 * (n - my_rank + 1) / n if n > 0 else None

        return best_algo, best_score, my_rank, n, my_percentile

    except Exception:
        return None, None, None, 0, None


# =========================
# 4) Programme principal
# =========================
def main():
    # --- CLI overrides ---
    # Support both `--visualization false` and Hydra-style `visualization=false` (e.g. via `-m`).
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--visualization", type=str, default=None,
                        help="Enable/disable visualization: true|false")
    parser.add_argument(
        "--grid-settings",
        type=str,
        default=None,
        help="Optional JSON/YAML file to override agent_M and agent_lambda in the sweep grid.",
    )
    args, _ = parser.parse_known_args()
    if args.visualization is not None:
        v = args.visualization.strip().lower()
        if v in ("0", "false", "no", "f"):
            DEFAULTS["visualization"] = False
        elif v in ("1", "true", "yes", "t"):
            DEFAULTS["visualization"] = True

    # Hydra-style overrides often appear as `visualization=false` in `sys.argv` (or after a `-m`).
    # If present, prefer these explicit overrides.
    for tok in sys.argv[1:]:
        if tok.startswith("visualization="):
            v = tok.split("=", 1)[1].strip().lower()
            if v in ("0", "false", "no", "f"):
                DEFAULTS["visualization"] = False
            elif v in ("1", "true", "yes", "t"):
                DEFAULTS["visualization"] = True
            break

    # Désactive la visu Tkinter par défaut hors de la machine archlinux
    hostname = socket.gethostname().lower()
    if "archlinux" not in hostname:
        DEFAULTS["visualization"] = False
        print(f"[INFO] hostname={hostname}: visualisation désactivée (Tkinter non utilisé).")

    # Grid overrides
    if args.grid_settings:
        overrides = _load_grid_settings(args.grid_settings)
        for key in ("agent_M", "agent_lambda"):
            if key in overrides:
                GRID[key] = list(overrides[key])
                print(f"[INFO] GRID override from {args.grid_settings}: {key}={GRID[key]}")
    # Seeds
    torch.manual_seed(DEFAULTS["seed"])
    np.random.seed(DEFAULTS["seed"])
    random.seed(DEFAULTS["seed"])

    # Répertoires
    script_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    agg_outdir = os.path.join(repo_root, "results", "aggregation")
    Path(agg_outdir).mkdir(parents=True, exist_ok=True)

    # Grille + filtres
    keys = list(GRID.keys())
    values = [GRID[k] for k in keys]
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*values)]

    # Filtres additionnels pour éliminer les combinaisons redondantes:
    #  - SVGD n'a pas d'effet quand M=1 ⇒ garder uniquement le pas par défaut
    #  - Pour REINFORCE, K_steps et delta_target sont ignorés ⇒ fixer une seule
    #    valeur représentative pour chaque paramètre.
    default_svgd = DEFAULTS.get("learning_rate_svgd", None)
    canonical_K = GRID.get("agent_K_steps", [0])[0]
    canonical_delta = GRID.get("agent_delta_target", [0.0])[0]

    filtered = []
    for cfg in combos:
        m_val = int(cfg.get("agent_M", 1))
        lr_svgd = float(cfg.get("agent_learning_rate_svgd", default_svgd or 0.0))
        if default_svgd is not None and m_val == 1 and abs(lr_svgd - default_svgd) > 1e-12:
            continue

        agent = str(cfg.get("agent", "ppo")).strip().lower()
        if agent == "reinforce":
            if int(cfg.get("agent_K_steps", canonical_K)) != canonical_K:
                continue
            if abs(float(cfg.get("agent_delta_target", canonical_delta)) - canonical_delta) > 1e-12:
                continue

        filtered.append(cfg)

    combos = filtered
    total = len(combos)

    # Accumulateurs pour overview en temps réel
    # - best par instance (min) pour les victoires
    # - avg par instance pour calculer winner_avg_score
    per_instance_best = dict()              # (problem, dim, type_instance, idx) -> (best_score, algo_key)
    per_inst_algo_best = defaultdict(dict)  # (problem,dim,type,inst) -> {algo_key: best_on_restarts}
    per_inst_algo_avg  = defaultdict(dict)  # (problem,dim,type,inst) -> {algo_key: avg_on_restarts}
    run_histories = {}  # (problem, dim, type_instance, algo_key) -> history

    # Boucle d’expérimentation
    for i, cfg in enumerate(combos, 1):
        device = DEFAULTS["device"]
        verbose = DEFAULTS["verbose"]  # True pour laisser UNE barre interne
        nb_instances_test = DEFAULTS["nb_instances_test"]
        nb_restarts = DEFAULTS["nb_restarts"]
        budget = DEFAULTS["budget"]
        lambda_ = int(cfg.get("agent_lambda", DEFAULTS["lambda_"]))
        typeStrategy = DEFAULTS["type_strategy"]
        type_problem = DEFAULTS["problem_name"]

        dim = int(cfg["problem_dim"])
        type_instance = int(cfg["problem_type_instance"])

        agent = str(cfg.get("agent", "ppo")).strip().lower()
        is_ppo = agent == "ppo"

        learning_rate = float(cfg["agent_learning_rate"])
        learning_rate_svgd = float(cfg.get("agent_learning_rate_svgd", DEFAULTS.get("learning_rate_svgd", 0.1)))
        M = int(cfg["agent_M"])
        K_steps = int(cfg["agent_K_steps"]) if is_ppo else 0
        delta_target = float(cfg["agent_delta_target"])
        if not is_ppo:
            delta_target = 0.0
        lambda_per_agent = (lambda_ / M) if M > 0 else float(lambda_)
        lambda_per_agent_str = f"{lambda_per_agent:.3f}".rstrip("0").rstrip(".")

        updateMethod = "PPO" if is_ppo else "REINFORCE"

        delta_disp = f"{cfg['agent_delta_target']:.4f}" if is_ppo else "n/a"
        k_disp = str(K_steps) if is_ppo else "n/a"
        print(
            f"=========================================================DEBUT=======================================================================\n"
            f"▶ Run {i}/{total} | agent={updateMethod} lr={learning_rate} K={k_disp} "
            f"delta={delta_disp} M={M} lr_svgd={learning_rate_svgd} "
            f"lambda/M={lambda_per_agent_str} | problem={type_problem} dim={dim} t={type_instance}"
        )

        # Préparation des chemins (instances)
        pathResult = os.path.join(
            repo_root, "results", "results_Multivariate-RL-EDA",
            typeStrategy, str(type_problem), str(dim), str(type_instance)
        ) + os.sep
        Path(pathResult).mkdir(parents=True, exist_ok=True)

        # Chargement instances
        if type_problem == "QUBO":
            instance_path = os.path.join(script_dir, "instances", "QUBO") + os.sep
            N = dim
            try:
                tensor_Q_test = getTensorInstances_QUBO(
                    instance_path, nb_instances_test, nb_restarts, N, type_instance, device, "test"
                )
            except FileNotFoundError:
                fallback_dim = 64
                print(f"[WARN] dim={N} indisponible; fallback dim={fallback_dim}")
                N = fallback_dim
                dim = fallback_dim
                pathResult = os.path.join(
                    repo_root, "results", "results_Multivariate-RL-EDA",
                    typeStrategy, str(type_problem), str(dim), str(type_instance)
                ) + os.sep
                Path(pathResult).mkdir(parents=True, exist_ok=True)
                tensor_Q_test = getTensorInstances_QUBO(
                    instance_path, nb_instances_test, nb_restarts, N, type_instance, device, "test"
                )
            dim_variables = None
            vectorIndex_th = None
            tensor_matrix_locus = None
            tensor_matrix_contrib = None
            D = None

        elif type_problem in ("NK", "NK3"):
            if type_problem == "NK":
                D = 2
                base_path = os.path.join(script_dir, "instances", "nk", str(dim), str(type_instance)) + os.sep
            else:
                D = 3
                base_path = os.path.join(script_dir, "instances", "nk3", str(dim), str(type_instance)) + os.sep

            vectorIndex = np.zeros((type_instance + 1))
            for vi in range(type_instance + 1):
                vectorIndex[vi] = D ** (type_instance - vi)
            vectorIndex_th = torch.tensor(vectorIndex, dtype=torch.float32).to(device)

            tensor_matrix_locus, tensor_matrix_contrib, tensor_Q_test = getTensorInstances_NK(
                base_path, nb_instances_test, nb_restarts, lambda_, dim, D, type_instance, device
            )

            if type_problem == "NK3":
                dim_variables = [3 for _ in range(dim)]
            else:
                dim_variables = None
        else:
            raise ValueError(f"type_problem inconnu: {type_problem}")

        # Fabrique de stratégie
        factory = FactoryStrategyEA()
        strategy = factory.createStrategyEA(
            typeStrategy,
            dim,
            lambda_,
            device,
            dim_variables,
            M,
            updateMethod=updateMethod,
            K_steps=K_steps,
            delta_target=delta_target,
            learning_rate=learning_rate,
            learning_rate_svgd=learning_rate_svgd,
            enable_visualization=DEFAULTS.get("visualization", True),
        ).to(device)

        # ---- Exécution avec chemin TEMPORAIRE (UNE barre), puis suppression immédiate ----
        t0 = time.time()
        run_history = None
        with tempfile.NamedTemporaryFile(prefix="rl_eda_", suffix=".log", delete=False) as tmpf:
            temp_path = tmpf.name
        try:
            if type_problem == "QUBO":
                list_scores, run_history = get_Score_trajectoriesQUBO_cuda(
                    strategy, dim, nb_instances_test, nb_restarts, budget, lambda_,
                    tensor_Q_test, device, verbose, temp_path,
                    enable_visualization=DEFAULTS.get("visualization", True),
                    return_history=True
                )
            elif type_problem in ("NK", "NK3"):
                list_scores = get_Score_trajectoriesNK_cuda(
                    strategy, dim, type_instance, D, nb_instances_test, nb_restarts, budget, lambda_,
                    vectorIndex_th, tensor_matrix_locus, tensor_matrix_contrib, device, verbose, temp_path
                )
            else:
                raise ValueError("Cas non prévu")
        finally:
            # on supprime systématiquement le fichier temporaire
            try:
                os.remove(temp_path)
            except OSError:
                pass
        dt = time.time() - t0

        # Moyenne globale du run (toutes instances × restarts)
        avg_score = float(np.mean(
            list_scores if isinstance(list_scores, (list, tuple))
            else (list_scores.detach().cpu().numpy() if torch.is_tensor(list_scores) else list_scores)
        ))
        if run_history is not None and "best_fitness" in run_history and run_history["best_fitness"]:
            # Aligne la dernière valeur de l'historique avec l'avg_score affiché (moyenne des scores renvoyés)
            run_history["best_fitness"][-1] = avg_score

        print(f"   ↳ avg_score={avg_score:.6f} | runtime={dt:.2f}s")

        # Affichage ranking pour ce run (optionnel)
        best_algo_csv, best_score_csv, my_rank, n_rank, my_pct = rank_vs_global_ranking(repo_root, dim, type_instance, avg_score)
        if best_algo_csv is not None:
            pct_str = f"{my_pct:.1f}%" if my_pct is not None else "n/a"
            print(f"   ↳ ranking file: best={best_algo_csv} ({best_score_csv:.2f}) | your avg rank: {my_rank}/{n_rank} ({pct_str})")
        else:
            print("   ↳ ranking file: introuvable/illisible pour ce groupe (pas d'affichage).")

        # Mise en forme par instance: [[r1..rR], [r1..rR], ...] len = nb_instances_test
        by_instance = flat_or_matrix_to_instances(list_scores, nb_instances_test, nb_restarts)

        # Mise à jour "meilleur algo par instance" (minimisation) + stockage des moyennes
        algo_key = (
            f"{updateMethod}:{DEFAULTS['type_strategy']}:lr{learning_rate}:K{K_steps}:"
            f"delta{delta_target}:M{M}:"
            f"lambdaPerAgent{lambda_per_agent_str}:lambdaTotal{lambda_}:lr_svgd{learning_rate_svgd}"
        )
        if run_history is not None:
            run_histories[(type_problem, dim, type_instance, algo_key)] = run_history

        for inst_idx, rest_scores in enumerate(by_instance):
            # MINIMISATION
            best_on_restarts = min(rest_scores) if rest_scores else float("+inf")
            avg_on_restarts  = float(np.mean(rest_scores)) if rest_scores else float("nan")
            inst_key = (type_problem, dim, type_instance, inst_idx)
            per_inst_algo_best[inst_key][algo_key] = best_on_restarts
            per_inst_algo_avg[inst_key][algo_key]  = avg_on_restarts
            prev = per_instance_best.get(inst_key, (float("+inf"), None))
            if best_on_restarts < prev[0]:
                per_instance_best[inst_key] = (best_on_restarts, algo_key)
        # ===== Mise à jour temps réel des agrégats (avec rank/percent) =====
        _write_realtime_aggregation(
            repo_root,
            agg_outdir,
            per_instance_best,
            per_inst_algo_best,
            per_inst_algo_avg,
            run_histories,
        )

    print("[DONE] Sweep terminé.")


def _load_existing_overview(path: str):
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


def _extract_best_fitness(values):
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


def _existing_history_best(csv_path: Path):
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


def _format_algo_display(algo_key: str) -> str:
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

    seen_lr_svgd = False
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
            seen_lr_svgd = True
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


def _write_realtime_aggregation(
    repo_root: str,
    agg_outdir: str,
    per_instance_best: dict,
    per_inst_algo_best: dict,
    per_inst_algo_avg: dict,
    run_histories: dict | None = None,
):
    """
    Écrit/Met à jour en temps réel:
      - best_algo_overview.csv (par groupe: winner + rank/percent + winner_avg_score)
      - best_algo_summary.txt  (lisible humain)

    **Winner = algo à la plus faible moyenne (winner_avg_score) sur les instances observées.**
    """
    # 1) Instances vues par groupe
    n_instances_seen = defaultdict(int)
    for (problem, dim, type_instance, inst_idx) in per_inst_algo_best.keys():
        n_instances_seen[(problem, dim, type_instance)] = max(
            n_instances_seen[(problem, dim, type_instance)],
            inst_idx + 1
        )

    # Regrouper toutes (problem, dim, type_instance)
    groups = sorted({(p, d, t) for (p, d, t, _) in per_inst_algo_avg.keys()},
                    key=lambda x: (x[0], x[1], x[2]))

    overview_csv = os.path.join(agg_outdir, "best_algo_overview.csv")
    existing_entries = _load_existing_overview(overview_csv)
    new_entries = {}
    history_by_group = {}

    for (problem, dim, type_instance) in groups:
        n_inst = n_instances_seen.get((problem, dim, type_instance), 0)
        # Agréger moyennes par algo
        algo_to_avgs = defaultdict(list)
        for inst_idx in range(n_inst):
            key_inst = (problem, dim, type_instance, inst_idx)
            for algo_key, avg_val in per_inst_algo_avg.get(key_inst, {}).items():
                if not (avg_val is None or np.isnan(avg_val)):
                    algo_to_avgs[algo_key].append(avg_val)

        if not algo_to_avgs:
            # Rien à écrire pour ce groupe
            continue

        # Moyenne globale par algo (sur les instances où on a des données)
        algo_mean = {
            algo: float(np.mean(vals)) for algo, vals in algo_to_avgs.items() if len(vals) > 0
        }
        if not algo_mean:
            continue

        # Winner = algo avec la plus faible moyenne (minimisation)
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
        winner_display = _format_algo_display(winner_algo)
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
    """
    Écrit l'historique (runtime, best_fitness, avg_hamming, avg_kl) pour le winner d'un groupe
    **seulement** si la fitness est meilleure que le fichier existant (minimisation).
    """
    if not history:
        return

    runtimes = history.get("runtime") or []
    best_fitness = history.get("best_fitness") or []
    avg_hamming = history.get("avg_hamming") or []
    avg_kl = history.get("avg_kl") or []

    length = max(len(runtimes), len(best_fitness), len(avg_hamming), len(avg_kl))
    if length == 0:
        return

    new_best = _extract_best_fitness(best_fitness)

    def _safe(arr, i):
        return arr[i] if i < len(arr) else ""

    safe_algo = algo_key.replace(":", "_").replace("/", "_")
    out_dir = Path(agg_outdir) / "instance_history"
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{problem}_{dim}_{type_instance}_"
    existing_files = list(out_dir.glob(f"{prefix}*.csv"))
    existing_best = None
    for f in existing_files:
        existing_best = _existing_history_best(f)
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
    lines = ["runtime,best_fitness,avg_hamming,avg_kl"]
    for i in range(length):
        lines.append(
            f"{_safe(runtimes, i)},{_safe(best_fitness, i)},{_safe(avg_hamming, i)},{_safe(avg_kl, i)}"
        )
    out_file.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
