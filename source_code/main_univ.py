#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
DEFAULTS = dict(
    device="cuda:0",
    seed=0,
    verbose=True,
    nb_instances_test=10,
    nb_restarts=10,
    budget=10000,
    lambda_=10,
    typeModel="NeuralNet",
    isUnivariate=1,
    numberHiddenLayersG=1,
    nh=20,
    knownIG=False,
    fixSamplingOrder=False,
    fixUpdateOrder=False,
    learnOrder=False,
    dropoutGen=0.0,
    dropoutTrain=0.0,
    withoutCausalMaskTraining=False,
    type_strategy="PPO-EDA",   # utilisé par la fabrique
    problem_name="QUBO",       # defaults: problem: qubo
    visualization=True,
)

# =========================
# 2) Grille d’hparams (comme hydra.sweeper.params)
# =========================
GRID = dict(
    agent=["ppo", "reinforce"],
    agent_learning_rate=[0.001, 0.003, 0.005, 0.008, 0.02, 0.01, 0.015],
    agent_M=[1],
    agent_K_steps=[4, 6, 8, 20, 10, 15],
    agent_Beta_adapt=[True, False],
    agent_beta=[0.5, 1.0],
    agent_delta_target=[0.0025, 0.006],
    problem_dim=[64, 128, 256],
    problem_type_instance=[0, 1, 2, 3, 4, 5],
)

# =========================
# 3) Filtres identiques à hydra_filter_sweeper.Expression
# =========================
def passes_filters(cfg):
    # 1) Si agent=reinforce ⇒ garder une seule combi (K_steps=2, Beta_adapt=False, delta_target=0.001)
    if cfg["agent"] == "reinforce":
        if not (cfg["agent_K_steps"] == 2 and cfg["agent_Beta_adapt"] is False and abs(cfg["agent_delta_target"] - 0.001) < 1e-12):
            return False
    # 2) Si Beta_adapt=True ⇒ beta doit être 1.0
    if cfg["agent_Beta_adapt"] is True and abs(cfg["agent_beta"] - 1.0) > 1e-12:
        return False
    # 3) Si Beta_adapt=False ⇒ delta_target doit être 0.001
    if cfg["agent_Beta_adapt"] is False and abs(cfg["agent_delta_target"] - 0.001) > 1e-12:
        return False
    return True


# =========================
# 4) Normalisation robuste des scores par instance
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
# 5) Programme principal
# =========================
def main():
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
    combos = [c for c in combos if passes_filters(c)]
    total = len(combos)

    # Accumulateurs pour overview en temps réel
    # - best par instance (min) pour les victoires
    # - avg par instance pour calculer winner_avg_score
    per_instance_best = dict()              # (problem, dim, type_instance, idx) -> (best_score, algo_key)
    per_inst_algo_best = defaultdict(dict)  # (problem,dim,type,inst) -> {algo_key: best_on_restarts}
    per_inst_algo_avg  = defaultdict(dict)  # (problem,dim,type,inst) -> {algo_key: avg_on_restarts}

    # Boucle d’expérimentation
    for i, cfg in enumerate(combos, 1):
        device = DEFAULTS["device"]
        verbose = DEFAULTS["verbose"]  # True pour laisser UNE barre interne
        nb_instances_test = DEFAULTS["nb_instances_test"]
        nb_restarts = DEFAULTS["nb_restarts"]
        budget = DEFAULTS["budget"]
        lambda_ = DEFAULTS["lambda_"]
        learnOrder = DEFAULTS["learnOrder"]
        dropoutGen = DEFAULTS["dropoutGen"]
        dropoutTrain = DEFAULTS["dropoutTrain"]
        withoutCausalMaskTraining = DEFAULTS["withoutCausalMaskTraining"]
        typeStrategy = DEFAULTS["type_strategy"]
        type_problem = DEFAULTS["problem_name"]

        dim = int(cfg["problem_dim"])
        type_instance = int(cfg["problem_type_instance"])

        # Agent params
        agent = cfg["agent"]  # "ppo" ou "reinforce"
        learning_rate = float(cfg["agent_learning_rate"])
        M = int(cfg["agent_M"])
        K_steps = int(cfg["agent_K_steps"])
        Beta_adapt = bool(cfg["agent_Beta_adapt"])
        beta_param = float(cfg["agent_beta"])
        delta_target = float(cfg["agent_delta_target"])

        # Conventions fabrique
        if agent.lower() == "ppo":
            updateMethod = "PPO"
        else:
            updateMethod = "REINFORCE"
            K_steps = 0
            Beta_adapt = False
            delta_target = 0.0

        print(
            f"=========================================================DEBUT=======================================================================\n"
            f"▶ Run {i}/{total} | agent={agent} lr={learning_rate} K={K_steps} "
            f"BetaAdapt={Beta_adapt} beta={beta_param} delta={delta_target} M={M} "
            f"| problem={type_problem} dim={dim} t={type_instance}"
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
            typeStrategy, dim, lambda_, beta_param, device,
            DEFAULTS["typeModel"], DEFAULTS["numberHiddenLayersG"], DEFAULTS["nh"],
            DEFAULTS["isUnivariate"], dropoutGen, dropoutTrain, withoutCausalMaskTraining,
            dim_variables, learnOrder, 1, M,
            updateMethod=updateMethod, K_steps=K_steps, beta_adapt=Beta_adapt,
            delta_target=delta_target, learning_rate=learning_rate
        )

        try:
            if torch.__version__ >= "2.0":
                strategy = torch.compile(strategy, mode="max-autotune")  # une seule fois
        except Exception as e:
            print("torch.compile désactivé:", e)

        # IG / ordres
        if DEFAULTS["knownIG"] and type_problem in ("QUBO", "NK", "NK3"):
            DAG = tensor_Q_test.unsqueeze(1).repeat(1, lambda_, 1, 1).to(device)
            DAG = torch.where(DAG != 0, 1, 0)
            strategy.setKnownDAG(DAG)

        if DEFAULTS["fixSamplingOrder"]:
            order = torch.tensor(np.arange(dim)).to(device)
            order = order.unsqueeze(0).unsqueeze(1).repeat(nb_instances_test * nb_restarts, lambda_, 1)
            strategy.setKnownOrder(order)

        if DEFAULTS["fixUpdateOrder"]:
            strategy.setSameDagTraining()

        # ---- Exécution avec chemin TEMPORAIRE (UNE barre), puis suppression immédiate ----
        t0 = time.time()
        with tempfile.NamedTemporaryFile(prefix="rl_eda_", suffix=".log", delete=False) as tmpf:
            temp_path = tmpf.name
        try:
            if type_problem == "QUBO":
                list_scores = get_Score_trajectoriesQUBO_cuda(
                    strategy, dim, nb_instances_test, nb_restarts, budget, lambda_,
                    tensor_Q_test, device, verbose, temp_path, enable_visualization=DEFAULTS.get("visualization", True)
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
        algo_key = f"{updateMethod}:{DEFAULTS['type_strategy']}:lr{learning_rate}:K{K_steps}:BetaAdapt{Beta_adapt}:beta{beta_param}:delta{delta_target}:M{M}"

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
            per_inst_algo_avg
        )

    print("[DONE] Sweep terminé.")


def _write_realtime_aggregation(
    repo_root: str,
    agg_outdir: str,
    per_instance_best: dict,
    per_inst_algo_best: dict,
    per_inst_algo_avg: dict
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

    # 2) Pour chaque groupe, calculer la moyenne par algo, puis choisir l'algo au score moyen minimal
    lines2 = ["problem,dim,type_instance,winner_algo,rank,percent,winner_avg_score"]
    human_lines = []

    # Regrouper toutes (problem, dim, type_instance)
    groups = sorted({(p, d, t) for (p, d, t, _) in per_inst_algo_avg.keys()},
                    key=lambda x: (x[0], x[1], x[2]))

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
        winner_algo = min(algo_mean.items(), key=lambda kv: kv[1])[0]
        winner_avg_score = algo_mean[winner_algo]

        # Rank & percent vs tableau global (sur winner_avg_score)
        _best_algo_csv, _best_score_csv, my_rank, n_rank, my_pct = rank_vs_global_ranking(
            repo_root, int(dim), int(type_instance), winner_avg_score
        )
        rank_val = my_rank if my_rank is not None else ""
        percent_str = f"{my_pct:.1f}" if my_pct is not None else ""

        lines2.append(f"{problem},{dim},{type_instance},{winner_algo},{rank_val},{percent_str},{winner_avg_score}")
        pct_disp = f"{percent_str}%" if percent_str != "" else "n/a"
        rank_disp = f"{rank_val}/{n_rank}" if (rank_val != "" and n_rank is not None) else "n/a"
        human_lines.append(
            f"{problem} | dim={dim} | type_instance={type_instance} -> "
            f"winner={winner_algo} | rank={rank_disp} ({pct_disp}) | winner_avg_score={winner_avg_score:.6f}"
        )

    overview_csv = os.path.join(agg_outdir, "best_algo_overview.csv")
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

    print("=========================================================FIN==========================================================================\n")


if __name__ == "__main__":
    main()
