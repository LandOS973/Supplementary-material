#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import datetime
import itertools
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import random

# --- imports projet (identiques à ton main) ---
from eda_strategies.FactoryStrategyEA import FactoryStrategyEA
from environment.qubo import getTensorInstances_QUBO, get_Score_trajectoriesQUBO_cuda
from environment.nk import getTensorInstances_NK, get_Score_trajectoriesNK_cuda
# ------------------------------------------------

"""
Script autonome qui balaie une grille d'hyperparamètres avec les MEMES FILTRES
que ton hydra/sweeper:filter, exécute les tests comme ton main, puis
écrit des logs finaux :
- results/aggregation/best_algo_per_instance.csv
- results/aggregation/best_algo_overview.csv  (avec winner_avg_best_score)
- results/aggregation/sweep_summary.json
- results/aggregation/best_algo_summary.txt
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
    problem_name="QUBO",       # comme defaults: problem: qubo
)

# =========================
# 2) Grille d’hparams (comme hydra.sweeper.params)
# =========================
GRID = dict(
    agent=["ppo", "reinforce"],
    agent_learning_rate=[0.01, 0.02, 0.03],
    agent_M=[1],
    agent_K_steps=[2, 4, 6, 8],
    agent_Beta_adapt=[True, False],
    agent_beta=[0.5, 1.0, 2.0],          # beta côté agent / EDA
    agent_delta_target=[0.001, 0.0025, 0.005],
    problem_dim=[64, 128, 256],
    problem_type_instance=[0, 1, 2, 3],
)

# =========================
# 3) Filtres équivalents à hydra_filter_sweeper.Expression
# =========================
def passes_filters(cfg):
    """
    Reproduit exactement:
      1) Si agent=reinforce ⇒ garder une seule combi (K_steps=2, Beta_adapt=False, delta_target=0.001)
      2) Si Beta_adapt=True ⇒ ne pas balayer beta (on fige beta=1.0)
      3) Si Beta_adapt=False ⇒ ne pas balayer delta_target (on fige delta_target=0.001)
    """
    agent = cfg["agent"]
    K_steps = cfg["agent_K_steps"]
    Beta_adapt = cfg["agent_Beta_adapt"]
    delta_target = cfg["agent_delta_target"]
    beta = cfg["agent_beta"]

    # 1)
    if agent == "reinforce":
        if not (K_steps == 2 and Beta_adapt is False and abs(delta_target - 0.001) < 1e-12):
            return False

    # 2)
    if Beta_adapt is True and abs(beta - 1.0) > 1e-12:
        return False

    # 3)
    if Beta_adapt is False and abs(delta_target - 0.001) > 1e-12:
        return False

    return True


# =========================
# 4) Normalisation robuste des scores par instance
# =========================
def flat_or_matrix_to_instances(list_scores, nb_instances, nb_restarts):
    """
    Convertit list_scores en matrice [nb_instances][nb_restarts].

    Formats supportés :
      - list[float] de taille nb_instances * nb_restarts
      - list[list[float]] de taille [nb_instances][nb_restarts]
      - numpy.ndarray 1D/2D (transpose si besoin)
      - torch.Tensor 1D/2D (transpose si besoin)
      - tuple/dict contenant le vrai tableau sous les clés/positions usuelles
        (ex: ('ok', arr), {'scores': arr}, {'list_scores': arr})
      - list[float] de taille nb_instances  (on suppose nb_restarts=1)

    En cas d’échec, lève ValueError avec un message qui affiche le type/shape.
    """
    import numpy as _np

    def _is_num(x):
        return isinstance(x, (int, float, _np.number))

    def _as_array(x):
        # Torch -> numpy
        try:
            import torch as _torch
            if isinstance(x, _torch.Tensor):
                x = x.detach().cpu().numpy()
        except Exception:
            pass
        # list/tuple -> numpy
        if isinstance(x, (list, tuple)):
            try:
                x = _np.array(x, dtype=float)
            except Exception:
                return x
        return x

    # Déballage si tuple/dict enveloppe
    candidate = list_scores
    if isinstance(candidate, tuple) and len(candidate) >= 1:
        if len(candidate) >= 2:
            maybe = _as_array(candidate[1])
            if isinstance(maybe, _np.ndarray):
                candidate = maybe
            else:
                candidate = _as_array(candidate[0])
        else:
            candidate = _as_array(candidate[0])
    elif isinstance(candidate, dict):
        for k in ("scores", "list_scores", "values", "arr", "data"):
            if k in candidate:
                candidate = _as_array(candidate[k])
                break

    # Première normalisation
    arr = _as_array(candidate)

    # Cas numpy array
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
            flat = arr.reshape(nb_instances, nb_restarts)
            return flat.tolist()
        raise ValueError(
            f"Format list_scores non supporté (numpy): shape={arr.shape}, "
            f"attendu (ni 1D len={nb_instances*nb_restarts} ni 2D {nb_instances}x{nb_restarts})."
        )

    # Cas liste/tuple Python
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
                out = []
                idx = 0
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

    raise ValueError(
        f"Format list_scores non supporté (type={t}). "
        f"Essayé numpy/list/tuple/torch 1D/2D. Aperçu={head}"
    )


# =========================
# 5) Programme principal
# =========================
def main():
    # ===== Seeds globaux =====
    torch.manual_seed(DEFAULTS["seed"])
    np.random.seed(DEFAULTS["seed"])
    random.seed(DEFAULTS["seed"])

    # ===== Répertoires =====
    script_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    agg_outdir = os.path.join(repo_root, "results", "aggregation")
    Path(agg_outdir).mkdir(parents=True, exist_ok=True)

    print("Création de la stratégie :", DEFAULTS["type_strategy"])

    # ===== Génération de la grille + filtres =====
    keys = list(GRID.keys())
    values = [GRID[k] for k in keys]

    combos = []
    for vals in itertools.product(*values):
        cfg = dict(zip(keys, vals))
        if passes_filters(cfg):
            combos.append(cfg)

    # ===== Accumulateurs =====
    per_instance_best = dict()            # key: (problem, dim, type_instance, idx) -> (best_score, algo_key)
    per_inst_algo_best = defaultdict(dict)  # key -> {algo_key: best_score}  (pour winner_avg_best_score)
    sweep_runs_summaries = []             # audit global des runs

    # ===== Boucle d’expérimentation =====
    for i, cfg in enumerate(combos, 1):
        device = DEFAULTS["device"]
        verbose = DEFAULTS["verbose"]
        nb_instances_test = DEFAULTS["nb_instances_test"]
        nb_restarts = DEFAULTS["nb_restarts"]
        budget = DEFAULTS["budget"]
        lambda_ = DEFAULTS["lambda_"]
        typeModel = DEFAULTS["typeModel"]
        isUnivariate = DEFAULTS["isUnivariate"]
        knownIG = DEFAULTS["knownIG"]
        fixSamplingOrder = DEFAULTS["fixSamplingOrder"]
        fixUpdateOrder = DEFAULTS["fixUpdateOrder"]
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

        # Conventions pour la fabrique (suivent ton main)
        if agent.lower() == "ppo":
            updateMethod = "PPO"
        else:
            updateMethod = "REINFORCE"
            # forçons la sémantique identique à ton main
            K_steps = 0
            Beta_adapt = False
            delta_target = 0.0

        # Préparation des chemins de résultats (comme ton main)
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

        # IG et ordres (comme ton main)
        if knownIG:
            if type_problem in ("QUBO", "NK", "NK3"):
                DAG = tensor_Q_test.unsqueeze(1).repeat(1, lambda_, 1, 1).to(device)
                DAG = torch.where(DAG != 0, 1, 0)
                strategy.setKnownDAG(DAG)
            else:
                print("IG unknown")

        if DEFAULTS["fixSamplingOrder"]:
            order = torch.tensor(np.arange(dim)).to(device)
            order = order.unsqueeze(0).unsqueeze(1).repeat(nb_instances_test * nb_restarts, lambda_, 1)
            strategy.setKnownOrder(order)

        if DEFAULTS["fixUpdateOrder"]:
            strategy.setSameDagTraining()

        # Nom de fichier résultat (comme ton main)
        nowstamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name_file_result = (
            "Test_"
            + DEFAULTS["type_strategy"] + "_" + type_problem
            + "_N_" + str(dim)
            + "_t_" + str(type_instance)
            + "_lambda_" + str(lambda_)
            + "_beta_" + str(beta_param)
            + "_typeModel_" + str(DEFAULTS["typeModel"])
            + "_learnOrder_" + str(learnOrder)
            + "_knownIG_" + str(knownIG)
            + "_fixSamplingOrder_" + str(DEFAULTS["fixSamplingOrder"])
            + "_fixUpdateOrder_" + str(DEFAULTS["fixUpdateOrder"])
            + "_L_" + str(DEFAULTS["numberHiddenLayersG"])
            + "_nh_" + str(DEFAULTS["nh"])
            + "_dGen_" + str(dropoutGen)
            + "_dTrain_" + str(dropoutTrain)
            + "_wCMaskTrain_" + str(withoutCausalMaskTraining)
            + "_" + nowstamp + "_" + str(DEFAULTS["seed"]) + ".txt"
        )

        # Exécution
        if type_problem == "QUBO":
            list_scores = get_Score_trajectoriesQUBO_cuda(
                strategy, dim, nb_instances_test, nb_restarts, budget, lambda_,
                tensor_Q_test, device, verbose, os.path.join(pathResult, name_file_result)
            )
        elif type_problem in ("NK", "NK3"):
            list_scores = get_Score_trajectoriesNK_cuda(
                strategy, dim, type_instance, D, nb_instances_test, nb_restarts,
                budget, lambda_, vectorIndex_th, tensor_matrix_locus,
                tensor_matrix_contrib, device, verbose, os.path.join(pathResult, name_file_result)
            )
        else:
            raise ValueError("Cas non prévu")

        # Debug format (une seule fois)
        if not hasattr(main, "_once_logged"):
            main._once_logged = True
            try:
                import numpy as _np
                import torch as _torch
                print("[DEBUG] type(list_scores):", type(list_scores))
                if isinstance(list_scores, (list, tuple)):
                    print("[DEBUG] len(list_scores):", len(list_scores))
                    if len(list_scores) > 0:
                        print("[DEBUG] type(list_scores[0]):", type(list_scores[0]))
                if isinstance(list_scores, _torch.Tensor):
                    print("[DEBUG] tensor shape:", tuple(list_scores.shape))
                if isinstance(list_scores, _np.ndarray):
                    print("[DEBUG] ndarray shape:", list_scores.shape)
            except Exception as e:
                print("[DEBUG] logging error:", e)

        avg_score = float(np.mean(
            list_scores if isinstance(list_scores, (list, tuple))
            else (list_scores.detach().cpu().numpy() if torch.is_tensor(list_scores) else list_scores)
        ))

        by_instance = flat_or_matrix_to_instances(list_scores, nb_instances_test, nb_restarts)

        # Mise à jour "meilleur algo par instance"
        algo_key = f"{updateMethod}:{DEFAULTS['type_strategy']}:lr{learning_rate}:K{K_steps}:BetaAdapt{Beta_adapt}:beta{beta_param}:delta{delta_target}:M{M}"
        for inst_idx, rest_scores in enumerate(by_instance):
            best_on_restarts = max(rest_scores) if rest_scores else float("-inf")
            key = (type_problem, dim, type_instance, inst_idx)

            # mémorise le meilleur score atteint par CET algo sur CETTE instance
            per_inst_algo_best[key][algo_key] = best_on_restarts

            # met à jour l'algo leader pour cette instance (comparaison stricte)
            prev = per_instance_best.get(key, (float("-inf"), None))
            if best_on_restarts > prev[0]:
                per_instance_best[key] = (best_on_restarts, algo_key)

        # Stockage résumé de run (audit)
        sweep_runs_summaries.append({
            "problem": type_problem,
            "dim": dim,
            "type_instance": type_instance,
            "nb_instances_test": nb_instances_test,
            "nb_restarts": nb_restarts,
            "budget": budget,
            "algo_key": algo_key,
            "agent": agent,
            "learning_rate": learning_rate,
            "M": M,
            "K_steps": K_steps,
            "Beta_adapt": Beta_adapt,
            "beta_param": beta_param,
            "delta_target": delta_target,
            "avg_score": avg_score,
            "result_file": name_file_result,
            "timestamp": nowstamp,
        })

        print(f"[{i}/{len(combos)}] {algo_key} | dim={dim} t={type_instance} -> avg={avg_score:.6f}")

    # ===== Écriture des logs finaux =====
    # 1) CSV par instance
    lines = ["problem,dim,type_instance,instance_idx,best_algo,best_score"]
    for key, val in sorted(per_instance_best.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
        (problem, dim, type_instance, inst_idx) = key
        best_score, best_algo = val
        lines.append(f"{problem},{dim},{type_instance},{inst_idx},{best_algo},{best_score}")
    per_instance_csv = os.path.join(agg_outdir, "best_algo_per_instance.csv")
    Path(per_instance_csv).write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Écrit: {per_instance_csv}")

    # 2) Vue overview (algo majoritaire par (problem, dim, type_instance)), avec moyenne du vainqueur
    tally = defaultdict(lambda: defaultdict(int))
    for key, val in per_instance_best.items():
        (problem, dim, type_instance, inst_idx) = key
        best_score, best_algo = val
        tally[(problem, dim, type_instance)][best_algo] += 1

    lines2 = ["problem,dim,type_instance,winner_algo,wins,n_instances,winner_avg_best_score"]
    human_lines = []

    for k, counter in sorted(tally.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        problem, dim, type_instance = k
        wins_total = sum(counter.values())
        winner_algo, wins = max(counter.items(), key=lambda kv: kv[1])

        # moyenne des meilleurs scores par instance obtenus par l'algo vainqueur
        scores_for_winner = []
        for inst_idx in range(DEFAULTS["nb_instances_test"]):
            key_inst = (problem, dim, type_instance, inst_idx)
            if winner_algo in per_inst_algo_best.get(key_inst, {}):
                scores_for_winner.append(per_inst_algo_best[key_inst][winner_algo])
        winner_avg = float(np.mean(scores_for_winner)) if scores_for_winner else float("nan")

        lines2.append(f"{problem},{dim},{type_instance},{winner_algo},{wins},{wins_total},{winner_avg}")

        human_lines.append(
            f"{problem} | dim={dim} | type_instance={type_instance} -> "
            f"winner={winner_algo} [{wins}/{wins_total}] | avg={winner_avg:.6f}"
        )

    overview_csv = os.path.join(agg_outdir, "best_algo_overview.csv")
    Path(overview_csv).write_text("\n".join(lines2), encoding="utf-8")
    print(f"[OK] Écrit: {overview_csv}")

    # 3) JSON complet
    summary_json = os.path.join(agg_outdir, "sweep_summary.json")
    Path(summary_json).write_text(json.dumps(sweep_runs_summaries, indent=2), encoding="utf-8")
    print(f"[OK] Écrit: {summary_json}")

    # 4) Résumé humain lisible
    pretty_txt = os.path.join(agg_outdir, "best_algo_summary.txt")
    header = [
        "=== BEST ALGO SUMMARY (par (problem, dim, type_instance)) ===",
        f"Généré le {datetime.datetime.now().isoformat(timespec='seconds')}",
        "",
        *human_lines,
        "",
        f"Détails par instance : {per_instance_csv}",
        f"Vue overview           : {overview_csv}",
        f"Résumé runs (JSON)     : {summary_json}",
        ""
    ]
    Path(pretty_txt).write_text("\n".join(header), encoding="utf-8")
    print(f"[OK] Écrit: {pretty_txt}")


if __name__ == "__main__":
    main()
