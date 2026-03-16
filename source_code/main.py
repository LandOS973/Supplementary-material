#!/usr/bin/env python3
"""
Simple runner for PPO-EDA (SVGD_EDA) using Hydra config.
No interactive prompts, just load config and run once.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from eda_strategies.FactoryStrategyEA import FactoryStrategyEA
from environment.blockwise import get_Score_trajectoriesBLOCK_cuda
from environment.nk import getTensorInstances_NK, get_Score_trajectoriesNK_cuda
from environment.qubo import getTensorInstances_QUBO, get_Score_trajectoriesQUBO_cuda
from environment.tsne_agents import plot_agents_tsne


def _load_kernel_config(kernel_name: str, repo_root: str) -> dict:
    kernel_dir = Path(repo_root) / "config" / "kernel"
    kernel_path = kernel_dir / f"{kernel_name}.yaml"
    if not kernel_path.exists():
        available = ", ".join(sorted(p.stem for p in kernel_dir.glob("*.yaml"))) if kernel_dir.exists() else "none"
        raise FileNotFoundError(
            f"Kernel config '{kernel_name}' introuvable dans {kernel_dir}. Kernels disponibles: {available}"
        )
    cfg = OmegaConf.load(str(kernel_path))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True) or {}
    if "name" not in cfg_dict:
        cfg_dict["name"] = kernel_name
    return cfg_dict


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
    M = int(agent_val("M") or cfg.get("M") or 1)

    kernel_name = str(agent_val("kernel") or cfg.get("kernel") or "hk").lower()
    kernel_cfg = _load_kernel_config(kernel_name, repo_root)
    prob_eps_override = agent_val("prob_eps_clamp") or cfg.get("prob_eps_clamp")
    if prob_eps_override is not None:
        kernel_cfg["prob_eps_clamp"] = float(prob_eps_override)
    natural_grad_override = agent_val("natural_grad") or cfg.get("natural_grad")
    if natural_grad_override is not None:
        kernel_cfg["natural_grad"] = bool(natural_grad_override)
    bandwith_override = agent_val("bandwith_kernel") or cfg.get("bandwith_kernel")
    if bandwith_override is not None:
        kernel_cfg["bandwith_kernel"] = bandwith_override

    kernel_lr = kernel_cfg.get("epsilon_svgd")
    kernel_gamma = kernel_cfg.get("gamma")
    epsilon_svgd = float(
        agent_val("epsilon_svgd")
        or cfg.get("epsilon_svgd")
        or kernel_lr
        or 0.5
    )
    svgd_gamma = float(
        agent_val("gamma")
        or cfg.get("gamma")
        or kernel_gamma
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

    print(
        f"Config: problem={type_problem} dim={dim} type_instance={type_instance} | "
        f"M={M} lambda={lambda_} eps={epsilon_svgd} gamma={svgd_gamma} | "
        f"kernel={kernel_name} advantage={advantage_cfg} decay={decay_enabled}"
    )
    if decay_enabled:
        print(f"Decay params: start_ratio={decay_start_ratio} min_factor={decay_min_factor}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    factory = FactoryStrategyEA()

    # Prepare problem-specific tensors
    dim_variables = None
    D = None
    block_size = None
    dummy_blocks = int(cfg.problem.dummy_blocks) if "problem" in cfg and "dummy_blocks" in cfg.problem else 0

    if type_problem == "QUBO":
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
    elif type_problem in ("NK", "NK3"):
        D = 2 if type_problem == "NK" else 3
        vectorIndex = np.zeros((type_instance + 1))
        for i in range(type_instance + 1):
            vectorIndex[i] = D ** (type_instance - i)
        vectorIndex_th = torch.tensor(vectorIndex, dtype=torch.float32).to(device)
        nk_path = os.path.join(script_dir, "instances", "nk" if type_problem == "NK" else "nk3",
                               str(dim), str(type_instance)) + os.sep
        tensor_matrix_locus, tensor_matrix_contrib, tensor_Q_test = getTensorInstances_NK(
            nk_path, nb_instances_test, nb_restarts, lambda_ * M, dim, D, type_instance, device
        )
    elif type_problem == "BLOCK":
        block_size = type_instance
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if dim % block_size != 0:
            raise ValueError(f"dim={dim} must be divisible by block_size={block_size}")
    else:
        raise ValueError(f"Unsupported problem type: {type_problem}")

    strategy = factory.createStrategyEA(
        "PPO-EDA",
        dim,
        lambda_,
        device,
        dim_variables,
        M,
        learning_rate=epsilon_svgd,
        epsilon_svgd=epsilon_svgd,
        enable_visualization=visualization_enabled,
        svgd_gamma=svgd_gamma,
        decay_start_ratio=decay_start_ratio,
        decay_min_factor=decay_min_factor,
        decay_enabled=decay_enabled,
        advantage_cfg=advantage_cfg,
        kernel_config=kernel_cfg,
        no_interact=no_interact,
        no_repulsion=no_repulsion,
    ).to(device)

    if type_problem == "QUBO":
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
            enable_visualization=visualization_enabled,
            return_history=False,
        )
        list_scores = result
    elif type_problem in ("NK", "NK3"):
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

    try:
        plot_agents_tsne(
            strategy,
            output_path=os.path.join(os.getcwd(), "agents_tsne.png"),
            perplexity=None,
            random_state=0,
        )
    except ValueError as exc:
        print(f"[WARN] t-SNE agents skipped: {exc}")


if __name__ == "__main__":
    main()
