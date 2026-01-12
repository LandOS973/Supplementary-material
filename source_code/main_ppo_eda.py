import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os
import random
from pathlib import Path
from eda_strategies.FactoryStrategyEA import FactoryStrategyEA
from environment.qubo import getTensorInstances_QUBO, get_Score_trajectoriesQUBO_cuda
from environment.blockwise import get_Score_trajectoriesBLOCK_cuda
from environment.nk import getTensorInstances_NK, get_Score_trajectoriesNK_cuda


import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True, formatter={"float_kind": lambda x: f"{x:.6f}"})

# Replication code for the article "Black-Box Combinatorial Optimization with Order-Invariant Reinforcement Learning"


def _load_kernel_config(kernel_name: str, repo_root: str):
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
        # propagate the requested kernel name so downstream components don't fall back to HK
        cfg_dict["name"] = kernel_name
    return cfg_dict


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):

    # Support keeping the original variable names used previously; read them from Hydra cfg
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('running on device: ' + device)
    type_problem = cfg.problem.name if 'problem' in cfg and 'name' in cfg.problem else cfg.get('type_problem', 'QUBO')
    print(f"Running with problem type: {type_problem}")
    dim = (
        cfg.problem.n
        if 'problem' in cfg and 'n' in cfg.problem
        else cfg.problem.dim if 'problem' in cfg and 'dim' in cfg.problem else cfg.get('dim', 64)
    )
    type_instance = (
        cfg.problem.k
        if 'problem' in cfg and 'k' in cfg.problem
        else cfg.problem.type_instance if 'problem' in cfg and 'type_instance' in cfg.problem else cfg.get('type_instance', 1)
    )
    print(f"Running with dim={dim}, type_instance={type_instance}")
    nb_restarts = int(cfg.nb_restarts)
    nb_instances_test = int(cfg.nb_instances_test)
    seed = int(cfg.seed)
    def agent_val(key):
        try:
            return OmegaConf.select(cfg, f"agent.{key}")
        except Exception:
            return None

    lambda_ = int(agent_val("lambda") or cfg.get('lambda') or cfg.get('lambda_') or 10)
    verbose = bool(cfg.get('verbose', True))
    budget = int(cfg.get('budget', 10000))
    visualization_enabled = bool(cfg.get('visualization', True))
    advantage_cfg = agent_val("advantage") or cfg.get('advantage') or "baseline"
    no_interact = bool(agent_val("no_interact") or cfg.get("no_interact") or False)
    if isinstance(advantage_cfg, DictConfig):
        advantage_cfg = OmegaConf.to_container(advantage_cfg, resolve=True)
    M = int(agent_val("M") or cfg.get('M') or 1)
    learning_rate = None
    typeStrategy = "PPO-EDA"
    script_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    kernel_name = str(agent_val("kernel") or cfg.get("kernel") or "hk").lower()
    kernel_cfg = _load_kernel_config(kernel_name, repo_root)
    kernel_lr = kernel_cfg.get("epsilon_svgd")
    kernel_gamma = kernel_cfg.get("gamma")
    kernel_bandwith_kernel = kernel_cfg.get("bandwith_kernel") or (kernel_cfg.get("params") or {}).get("bandwith_kernel")
    epsilon_svgd = float(
        agent_val("epsilon_svgd")
        or cfg.get('epsilon_svgd')
        or kernel_lr
        or 0.5
    )
    learning_rate = epsilon_svgd
    svgd_gamma = float(
        agent_val("gamma")
        or cfg.get('gamma')
        or kernel_gamma
        or 10.0
    )
    bandwith_kernel_suffix = ""
    if kernel_name in ("pk", "rbf"):
        bandwith_kernel_suffix = f", bandwith_kernel: {kernel_bandwith_kernel}"
    print(
        f"Using REINFORCE update. Number of agents: {M} with epsilon_svgd: {epsilon_svgd}, "
        f"λ: {lambda_}, svgd_gamma: {svgd_gamma}, advantage={advantage_cfg}, "
        f"kernel={kernel_name}{bandwith_kernel_suffix}, no_interact={no_interact}, bandwith_kernel: {kernel_bandwith_kernel}"
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    N = dim

    write_logs = bool(cfg.get('write_logs', False))
    pathResult = None

    block_size = None
    dummy_blocks = 0
    if (type_problem == "QUBO"):

        # Instances live under source_code/instances in this repo; resolve absolute path
        # add trailing sep because downstream loader concatenates filenames
        instance_path = os.path.join(script_dir, "instances", "QUBO") + os.sep
        try:
            tensor_Q_test = getTensorInstances_QUBO(instance_path, nb_instances_test, nb_restarts, N, type_instance, device,
                                                    "test")
        except FileNotFoundError as e:
            # fallback to a default dimension if requested instances not available
            fallback_dim = 64
            print(f"Requested problem dim={N} not available; falling back to default dim={fallback_dim}.")
            N = fallback_dim
            dim = fallback_dim
            # recompute pathResult for fallback dim
            if write_logs:
                pathResult = os.path.join(repo_root, "results", "results_Multivariate-RL-EDA", typeStrategy, str(type_problem), str(dim), str(type_instance)) + os.sep
                os.makedirs(pathResult, exist_ok=True)
            tensor_Q_test = getTensorInstances_QUBO(instance_path, nb_instances_test, nb_restarts, N, type_instance, device,
                                                    "test")
    elif(type_problem == "NK"):

        D = 2
        vectorIndex = np.zeros((type_instance + 1))
        for i in range(type_instance + 1):
            vectorIndex[i] = D ** (type_instance - i)
        vectorIndex_th = torch.tensor(vectorIndex, dtype=torch.float32).to(device)

        nk_path = os.path.join(script_dir, "instances", "nk", str(dim), str(type_instance)) + os.sep
        tensor_matrix_locus, tensor_matrix_contrib, tensor_Q_test = getTensorInstances_NK(nk_path, nb_instances_test, nb_restarts, lambda_, dim, D, type_instance, device)

    elif(type_problem == "NK3"):

        D = 3
        vectorIndex = np.zeros((type_instance + 1))
        for i in range(type_instance + 1):
            vectorIndex[i] = D ** (type_instance - i)
        vectorIndex_th = torch.tensor(vectorIndex, dtype=torch.float32).to(device)

        nk3_path = os.path.join(script_dir, "instances", "nk3", str(dim), str(type_instance)) + os.sep
        tensor_matrix_locus, tensor_matrix_contrib, tensor_Q_test = getTensorInstances_NK(nk3_path, nb_instances_test, nb_restarts, lambda_, dim, D, type_instance, device)
    elif type_problem == "BLOCK":
        block_size = type_instance
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if N % block_size != 0:
            raise ValueError(f"dim={N} must be divisible by block_size={block_size}")
        dummy_blocks = int(cfg.problem.dummy_blocks) if "problem" in cfg and "dummy_blocks" in cfg.problem else 0



    factory = FactoryStrategyEA()


    if (type_problem == "NK3"):
        dim_variables = [3 for i in range(N)]
    else:
        dim_variables = None


    strategy = factory.createStrategyEA(
        typeStrategy,
        dim,
        lambda_,
        device,
        dim_variables,
        M,
        learning_rate=learning_rate,
        epsilon_svgd=epsilon_svgd,
        enable_visualization=visualization_enabled,
        svgd_gamma=svgd_gamma,
        advantage_cfg=advantage_cfg,
        kernel_config=kernel_cfg,
        no_interact=no_interact,
    ).to(device)
    if (type_problem == "QUBO"):
        list_scores = get_Score_trajectoriesQUBO_cuda(
            strategy,
            N,
            nb_instances_test,
            nb_restarts,
            budget,
            lambda_,
            tensor_Q_test,
            device,
            verbose,
            enable_visualization=visualization_enabled,
        )

    elif (type_problem == "NK" or type_problem == "NK3"):
        list_scores = get_Score_trajectoriesNK_cuda(strategy, N,  type_instance, D, nb_instances_test, nb_restarts, 
                                                    budget, lambda_,
                                                    vectorIndex_th, tensor_matrix_locus,
                                                    tensor_matrix_contrib, device, verbose)
    elif type_problem == "BLOCK":
        list_scores = get_Score_trajectoriesBLOCK_cuda(
            strategy,
            N,
            block_size,
            nb_instances_test,
            nb_restarts,
            budget,
            lambda_,
            device,
            verbose,
            enable_visualization=visualization_enabled,
            dummy_blocks=dummy_blocks,
        )
        
    print(list_scores)
    average_test_score = np.mean(list_scores)

    print("average_test_score : " + str(average_test_score))

if __name__ == '__main__':
    # Run hydra main
    main()
