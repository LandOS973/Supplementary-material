"""
Simple runner for PPO-EDA (SVGD_EDA) using Hydra config.
No interactive prompts, just load config and run once.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from eda_strategies.FactoryStrategyEA import FactoryStrategyEA
from environment.blockwise import get_Score_trajectoriesBLOCK_cuda
from environment.nk import getTensorInstances_NK, get_Score_trajectoriesNK_cuda
from environment.qubo import getTensorInstances_QUBO, get_Score_trajectoriesQUBO_cuda
from main_viennarna import (
    DEFAULT_TARGET_NAME,
    DEFAULT_TARGET_STRUCT,
    RNA as VIENNA_RNA,
    get_Score_trajectories_viennarna_cuda,
)
from problems.viennarna import ETERNA100_TSV_URL, load_target_from_eterna100, normalize_target_struct
from utils.main_utils import build_global_ranking_lines


def print_global_ranking(
    repo_root: str, type_problem: str, dim: int, type_instance: int, avg_score: float, target_name: str | None = None
) -> None:
    """Situe notre score parmi les algos de additional_results/global_ranking (top 1, +-2 autour de nous)."""
    lines = build_global_ranking_lines(repo_root, type_problem, dim, type_instance, avg_score, target_name=target_name)
    print("\n=== " + lines[0] + " ===")
    for line in lines[1:]:
        print(line)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"running on device: {device}")

    script_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))

    def agent_val(key):
        try:
            return OmegaConf.select(cfg, f"agent.{key}")
        except Exception:
            return None

    type_problem = cfg.problem.name if "problem" in cfg and "name" in cfg.problem else cfg.get("type_problem", "QUBO")
    type_problem = str(type_problem)
    type_problem_upper = type_problem.upper()
    type_problem_lower = type_problem.lower()
    dim = (
        cfg.problem.n
        if "problem" in cfg and "n" in cfg.problem
        else cfg.problem.dim if "problem" in cfg and "dim" in cfg.problem else cfg.get("dim", 64)
    )
    type_instance = (
        cfg.problem.k
        if "problem" in cfg and "k" in cfg.problem
        else cfg.problem.type_instance if "problem" in cfg and "type_instance" in cfg.problem else cfg.get("type_instance", 1)
    )

    nb_restarts = int(cfg.nb_restarts)
    nb_instances_test = int(cfg.nb_instances_test)
    seed = int(cfg.seed)
    lambda_ = int(agent_val("lambda") or cfg.get("lambda") or cfg.get("lambda_") or 10)
    verbose = bool(cfg.get("verbose", True))
    budget = int(cfg.get("budget", 10000))
    visualization_enabled = bool(cfg.get("visualization", True))
    advantage_cfg = agent_val("advantage") or cfg.get("advantage") or "baseline"
    if isinstance(advantage_cfg, DictConfig):
        advantage_cfg = OmegaConf.to_container(advantage_cfg, resolve=True)
    no_interact = bool(agent_val("no_interact") or cfg.get("no_interact") or False)
    no_repulsion = bool(agent_val("no_repulsion") or cfg.get("no_repulsion") or False)
    decay_enabled = bool(agent_val("decay") or cfg.get("decay") or False)
    enable_greedy_final = agent_val("enable_greedy_final")
    if enable_greedy_final is None:
        enable_greedy_final = cfg.get("enable_greedy_final", True)
    enable_greedy_final = bool(enable_greedy_final)
    M = int(agent_val("M") or cfg.get("M") or 1)

    kernel_name = str(agent_val("kernel") or cfg.get("kernel") or "rbf").lower()
    prob_eps_override = agent_val("prob_eps_clamp") or cfg.get("prob_eps_clamp")
    prob_eps_clamp = float(prob_eps_override) if prob_eps_override is not None else 1e-3
    epsilon_svgd = float(
        agent_val("epsilon_svgd")
        or cfg.get("epsilon_svgd")
        or 0.5
    )
    svgd_gamma = float(
        agent_val("gamma")
        or cfg.get("gamma")
        or 10.0
    )

    decay_default_start_ratio = 0.0 if decay_enabled else 0.8
    decay_default_min_factor = 0.05 if decay_enabled else 0.1
    decay_start_ratio = float(
        agent_val("decay_start_ratio")
        or cfg.get("decay_start_ratio")
        or decay_default_start_ratio
    )
    decay_min_factor = float(
        agent_val("min_factor")
        or agent_val("decay_min_factor")
        or cfg.get("min_factor")
        or cfg.get("decay_min_factor")
        or decay_default_min_factor
    )
    ppo_active_val = agent_val("ppo_active")
    if ppo_active_val is None:
        ppo_active_val = cfg.get("ppo_active")
    ppo_active = bool(ppo_active_val) if ppo_active_val is not None else False

    kl_beta = 1.0

    if ppo_active:
        ppo_epochs = int(agent_val("ppo.ppo_epochs") or 4)
        kl_beta = float(agent_val("ppo.beta") or 1.0)
    else:
        ppo_epochs = 1

    if not ppo_active:
        ppo_info = "ppo_active=False"
    else:
        ppo_info = f"ppo_active=True mode=kl ppo_epochs={ppo_epochs} beta={kl_beta}"
    print(
        f"Config: problem={type_problem} dim={dim} type_instance={type_instance} | "
        f"M={M} lambda={lambda_} eps={epsilon_svgd} gamma={svgd_gamma} | "
        f"kernel={kernel_name} advantage={advantage_cfg} decay={decay_enabled} "
        f"greedy_final={enable_greedy_final} | "
        f"{ppo_info}"
    )
    if decay_enabled:
        print(f"Decay params: start_ratio={decay_start_ratio} min_factor={decay_min_factor}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    factory = FactoryStrategyEA()

    max_dim = None
    D = None
    block_size = None
    dummy_blocks = int(cfg.problem.dummy_blocks) if "problem" in cfg and "dummy_blocks" in cfg.problem else 0

    target_struct = None
    target_resolved_name = None
    num_workers = None

    if type_problem_upper == "QUBO":
        instance_path = os.path.join(script_dir, "instances", "QUBO") + os.sep
        try:
            tensor_Q_test = getTensorInstances_QUBO(
                instance_path, nb_instances_test, nb_restarts, dim, type_instance, device, "test"
            )
        except FileNotFoundError:
            fallback_dim = 64
            print(f"[WARN] dim={dim} indisponible; fallback dim={fallback_dim}")
            dim = fallback_dim
            tensor_Q_test = getTensorInstances_QUBO(
                instance_path, nb_instances_test, nb_restarts, dim, type_instance, device, "test"
            )
    elif type_problem_upper in ("NK", "NK3"):
        D = 2 if type_problem_upper == "NK" else 3
        vectorIndex = np.zeros((type_instance + 1))
        for i in range(type_instance + 1):
            vectorIndex[i] = D ** (type_instance - i)
        vectorIndex_th = torch.tensor(vectorIndex, dtype=torch.float32).to(device)
        nk_path = os.path.join(script_dir, "instances", "nk" if type_problem_upper == "NK" else "nk3",
                               str(dim), str(type_instance)) + os.sep
        tensor_matrix_locus, tensor_matrix_contrib, tensor_Q_test = getTensorInstances_NK(
            nk_path, nb_instances_test, nb_restarts, lambda_ * M, dim, D, type_instance, device
        )
        if type_problem_upper == "NK3":
            max_dim = 3
    elif type_problem_upper == "BLOCK":
        block_size = type_instance
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if dim % block_size != 0:
            raise ValueError(f"dim={dim} must be divisible by block_size={block_size}")
    elif type_problem_upper == "VIENNARNA":
        if VIENNA_RNA is None:
            raise RuntimeError(
                "Import `RNA` failed. Install ViennaRNA Python bindings first "
                "(e.g. `pip install ViennaRNA` or your `officievienna` package). "
                "If build fails, install SWIG and ViennaRNA development headers."
            )
        target_name = str(OmegaConf.select(cfg, "problem.target_name") or DEFAULT_TARGET_NAME)
        target_source = str(OmegaConf.select(cfg, "problem.target_source") or ETERNA100_TSV_URL)
        target_struct_cfg = OmegaConf.select(cfg, "problem.target_struct")
        if target_struct_cfg:
            target_struct = normalize_target_struct(str(target_struct_cfg))
            target_resolved_name = target_name
            print(f"[ViennaRNA] using target from cfg.problem.target_struct (len={len(target_struct)})")
        else:
            target_struct, target_resolved_name = load_target_from_eterna100(
                target_name=target_name,
                source=target_source,
                fallback_target=DEFAULT_TARGET_STRUCT,
                verbose=bool(cfg.get("verbose", True)),
            )
            print(f"[ViennaRNA] loaded target={target_resolved_name} (len={len(target_struct)})")
        dim = len(target_struct)
        max_dim = 4
        num_workers_cfg = OmegaConf.select(cfg, "problem.num_workers")
        if num_workers_cfg is None:
            num_workers_cfg = cfg.get("num_workers")
        num_workers = max(1, int(num_workers_cfg)) if num_workers_cfg is not None else max(1, mp.cpu_count() - 1)
    else:
        raise ValueError(f"Unsupported problem type: {type_problem}")

    strategy = factory.createStrategyEA(
        "PPO-EDA",
        dim,
        lambda_,
        device,
        max_dim,
        M,
        epsilon_svgd=epsilon_svgd,
        enable_visualization=visualization_enabled,
        svgd_gamma=svgd_gamma,
        decay_start_ratio=decay_start_ratio,
        decay_min_factor=decay_min_factor,
        decay_enabled=decay_enabled,
        advantage_cfg=advantage_cfg,
        kernel_name=kernel_name,
        prob_eps_clamp=prob_eps_clamp,
        no_interact=no_interact,
        no_repulsion=no_repulsion,
        ppo_active=ppo_active,
        ppo_epochs=ppo_epochs,
        kl_beta=kl_beta,
    ).to(device)
    if not enable_greedy_final:
        strategy.sample_greedy_agent_solutions = None

    if type_problem_upper == "QUBO":
        result = get_Score_trajectoriesQUBO_cuda(
            strategy,
            dim,
            nb_instances_test,
            nb_restarts,
            budget,
            lambda_,
            tensor_Q_test,
            device,
            verbose,
            type_instance=type_instance,
            enable_visualization=visualization_enabled,
            return_history=False,
        )
        list_scores = result
    elif type_problem_upper in ("NK", "NK3"):
        result = get_Score_trajectoriesNK_cuda(
            strategy,
            dim,
            type_instance,
            D,
            nb_instances_test,
            nb_restarts,
            budget,
            lambda_,
            vectorIndex_th,
            tensor_matrix_locus,
            tensor_matrix_contrib,
            device,
            verbose,
            enable_visualization=visualization_enabled,
            return_history=False,
        )
        list_scores = result
    elif type_problem_upper == "VIENNARNA":
        result = get_Score_trajectories_viennarna_cuda(
            target_struct=target_struct,
            strategy=strategy,
            nb_instances=nb_instances_test,
            budget=budget,
            size_popEA=lambda_,
            device=device,
            verbose=verbose,
            num_workers=num_workers,
            enable_visualization=visualization_enabled,
            name_file=None,
            return_history=False,
        )
        list_scores = result
    else:
        result = get_Score_trajectoriesBLOCK_cuda(
            strategy,
            dim,
            block_size,
            nb_instances_test,
            nb_restarts,
            budget,
            lambda_,
            device,
            verbose,
            enable_visualization=visualization_enabled,
            dummy_blocks=dummy_blocks,
            return_history=False,
        )
        list_scores = result

    avg = float(np.mean(list_scores))
    print("average_test_score:", avg)
    print_global_ranking(repo_root, type_problem, dim, type_instance, avg, target_name=target_resolved_name)


if __name__ == "__main__":
    main()
