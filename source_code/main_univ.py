#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test commit depuis nautilus
import os
import itertools
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import random
import argparse
import sys
import socket

# --- imports projet (identiques à ton main) ---
from eda_strategies.FactoryStrategyEA import FactoryStrategyEA
from environment.qubo import getTensorInstances_QUBO, get_Score_trajectoriesQUBO_cuda
from environment.blockwise import get_Score_trajectoriesBLOCK_cuda
from environment.nk import getTensorInstances_NK, get_Score_trajectoriesNK_cuda
from utils.main_utils import (
    flat_or_matrix_to_instances,
    load_grid_settings,
    rank_vs_global_ranking,
    write_realtime_aggregation,
)
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
# 1) Defaults 
# =========================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

DEFAULTS = dict(
    device=device,
    seed=0,
    verbose=True,
    nb_instances_test=10,
    nb_restarts=10,
    budget=50000,
    agent_lambda=10,
    type_strategy="PPO-EDA",   # utilisé par la fabrique
    problem_name="QUBO",       # defaults: problem: qubo
    visualization=False,
    agent_epsilon_svgd=0.2,
    svgd_gamma=10.0,
    advantage="baseline",
)

# =========================
# 2) Grille d’hparams 
# =========================
GRID = dict(
    agent_M=[1, 2, 4, 5],
    agent_epsilon_svgd=[0.2, 0.5, 1.0],
    problem_dim=[64, 128, 256],
    problem_type_instance=[0, 1, 2, 3, 4, 5],
    agent_lambda=[10, 15, 20, 25],
)

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
        overrides = load_grid_settings(args.grid_settings)
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
    default_svgd = DEFAULTS.get("agent_epsilon_svgd", None)

    filtered = []
    for cfg in combos:
        m_val = int(cfg.get("agent_M", 1))
        lr_svgd = float(cfg.get("agent_epsilon_svgd", default_svgd or 0.0))
        if default_svgd is not None and m_val == 1 and abs(lr_svgd - default_svgd) > 1e-12:
            continue

        filtered.append(cfg)

    combos = filtered
    total = len(combos)

    # Accumulateurs pour overview en temps réel
    # - avg par instance pour calculer winner_avg_score
    per_inst_algo_avg  = defaultdict(dict)  # (problem,dim,type,inst) -> {algo_key: avg_on_restarts}
    run_histories = {}  # (problem, dim, type_instance, algo_key) -> history

    # Boucle d’expérimentation
    for i, cfg in enumerate(combos, 1):
        device = DEFAULTS["device"]
        verbose = DEFAULTS["verbose"]  # True pour laisser UNE barre interne
        nb_instances_test = DEFAULTS["nb_instances_test"]
        nb_restarts = DEFAULTS["nb_restarts"]
        budget = DEFAULTS["budget"]
        lambda_ = int(cfg.get("agent_lambda", DEFAULTS["agent_lambda"]))
        typeStrategy = DEFAULTS["type_strategy"]
        type_problem = DEFAULTS["problem_name"]

        dim = int(cfg["problem_dim"])
        type_instance = int(cfg["problem_type_instance"])

        epsilon_svgd = float(
            cfg.get("agent_epsilon_svgd", DEFAULTS.get("agent_epsilon_svgd", 0.1))
        )
        learning_rate = epsilon_svgd
        svgd_gamma = float(
            cfg.get("svgd_gamma", DEFAULTS.get("svgd_gamma", 10.0))
        )
        advantage_cfg = cfg.get("advantage", DEFAULTS.get("advantage", "baseline"))
        M = int(cfg["agent_M"])
        lambda_per_agent = (lambda_ / M) if M > 0 else float(lambda_)
        lambda_per_agent_str = f"{lambda_per_agent:.3f}".rstrip("0").rstrip(".")

        print(
            f"=========================================================DEBUT=======================================================================\n"
            f"▶ Run {i}/{total} | agent=REINFORCE epsilon_svgd={epsilon_svgd} "
            f"M={M} lr_svgd={epsilon_svgd} "
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
        elif type_problem == "BLOCK":
            block_size = type_instance
            if block_size <= 0:
                raise ValueError(f"block_size must be positive, got {block_size}")
            if dim % block_size != 0:
                raise ValueError(f"dim={dim} must be divisible by block_size={block_size}")
            dim_variables = None
            vectorIndex_th = None
            tensor_matrix_locus = None
            tensor_matrix_contrib = None
            D = None
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
            learning_rate=learning_rate,
            epsilon_svgd=epsilon_svgd,
            enable_visualization=DEFAULTS.get("visualization", True),
            svgd_gamma=svgd_gamma,
            advantage_cfg=advantage_cfg,
            kernel_config=None,
        ).to(device)

        # ---- Exécution avec chemin TEMPORAIRE (UNE barre), puis suppression immédiate ----
        t0 = time.time()
        run_history = None
        if type_problem == "QUBO":
            list_scores, run_history = get_Score_trajectoriesQUBO_cuda(
                strategy, dim, nb_instances_test, nb_restarts, budget, lambda_,
                tensor_Q_test, device, verbose,
                enable_visualization=DEFAULTS.get("visualization", True),
                return_history=True
            )
        elif type_problem in ("NK", "NK3"):
            list_scores, run_history = get_Score_trajectoriesNK_cuda(
                strategy, dim, type_instance, D, nb_instances_test, nb_restarts, budget, lambda_,
                vectorIndex_th, tensor_matrix_locus, tensor_matrix_contrib, device, verbose,
                enable_visualization=DEFAULTS.get("visualization", True),
                return_history=True
            )
        elif type_problem == "BLOCK":
            list_scores, run_history = get_Score_trajectoriesBLOCK_cuda(
                strategy,
                dim,
                block_size,
                nb_instances_test,
                nb_restarts,
                budget,
                lambda_,
                device,
                verbose,
                enable_visualization=DEFAULTS.get("visualization", True),
                return_history=True,
            )
        else:
            raise ValueError("Cas non prévu")
        dt = time.time() - t0

        # Moyenne globale du run (toutes instances × restarts)
        avg_score = float(np.mean(
            list_scores if isinstance(list_scores, (list, tuple))
            else (list_scores.detach().cpu().numpy() if torch.is_tensor(list_scores) else list_scores)
        ))
        if run_history is not None and "best_fitness" in run_history and run_history["best_fitness"]:
            # Aligne la dernière valeur de l'historique avec l'avg_score affiché (moyenne des scores renvoyés)
            run_history["best_fitness"][-1] = avg_score

        print(f"   ↳ best_score={avg_score:.6f} | runtime={dt:.2f}s")

        # Affichage ranking pour ce run (optionnel)
        best_algo_csv, best_score_csv, my_rank, n_rank, my_pct = rank_vs_global_ranking(
            repo_root, type_problem, dim, type_instance, avg_score
        )
        if best_algo_csv is not None:
            pct_str = f"{my_pct:.1f}%" if my_pct is not None else "n/a"
            print(f"   ↳ ranking file: best={best_algo_csv} ({best_score_csv:.2f}) | your avg rank: {my_rank}/{n_rank} ({pct_str})")
        else:
            print("   ↳ ranking file: introuvable/illisible pour ce groupe (pas d'affichage).")

        # Mise en forme par instance: [[r1..rR], [r1..rR], ...] len = nb_instances_test
        by_instance = flat_or_matrix_to_instances(list_scores, nb_instances_test, nb_restarts)

        # Mise à jour "meilleur algo par instance" (minimisation) + stockage des moyennes
        if isinstance(advantage_cfg, str):
            advantage_type = advantage_cfg
        elif isinstance(advantage_cfg, dict):
            advantage_type = advantage_cfg.get("type", "baseline")
        else:
            advantage_type = "baseline"
        algo_key = (
            f"REINFORCE:{DEFAULTS['type_strategy']}:epsilon_svgd{epsilon_svgd}:"
            f"M{M}:lambdaPerAgent{lambda_per_agent_str}:lambdaTotal{lambda_}:lr_svgd{epsilon_svgd}"
            f":scoreDiv{svgd_gamma}:advantage{advantage_type}"
        )
        if run_history is not None:
            run_histories[(type_problem, dim, type_instance, algo_key)] = run_history

        for inst_idx, rest_scores in enumerate(by_instance):
            # MINIMISATION
            avg_on_restarts  = float(np.mean(rest_scores)) if rest_scores else float("nan")
            inst_key = (type_problem, dim, type_instance, inst_idx)
            per_inst_algo_avg[inst_key][algo_key]  = avg_on_restarts
        # ===== Mise à jour temps réel des agrégats (avec rank/percent) =====
        write_realtime_aggregation(
            repo_root,
            agg_outdir,
            per_inst_algo_avg,
            run_histories,
        )

    print("[DONE] Sweep terminé.")


if __name__ == "__main__":
    main()
