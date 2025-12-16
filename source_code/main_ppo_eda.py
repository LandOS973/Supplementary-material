import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os
import random
from pathlib import Path
from eda_strategies.FactoryStrategyEA import FactoryStrategyEA
from environment.qubo import getTensorInstances_QUBO, get_Score_trajectoriesQUBO_cuda
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
    return OmegaConf.to_container(cfg, resolve=True)


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):

    # Support keeping the original variable names used previously; read them from Hydra cfg
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('running on device: ' + device)
    type_problem = cfg.problem.name if 'problem' in cfg and 'name' in cfg.problem else cfg.get('type_problem', 'QUBO')
    print(f"Running with problem type: {type_problem}")
    dim = cfg.problem.dim if 'problem' in cfg and 'dim' in cfg.problem else cfg.get('dim', 64)
    type_instance = cfg.problem.type_instance if 'problem' in cfg and 'type_instance' in cfg.problem else cfg.get('type_instance', 1)
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
    if isinstance(advantage_cfg, DictConfig):
        advantage_cfg = OmegaConf.to_container(advantage_cfg, resolve=True)
    M = int(agent_val("M") or cfg.get('M') or 1)
    learning_rate = float(agent_val("learning_rate") or cfg.get('learning_rate') or 0.0)
    typeStrategy = "PPO-EDA"
    script_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    kernel_name = str(agent_val("kernel") or cfg.get("kernel") or "hk").lower()
    kernel_cfg = _load_kernel_config(kernel_name, repo_root)
    kernel_lr = kernel_cfg.get("learning_rate_svgd")
    kernel_alpha = kernel_cfg.get("alpha")
    learning_rate_svgd = float(
        agent_val("learning_rate_svgd")
        or cfg.get('learning_rate_svgd')
        or kernel_lr
        or 0.5
    )
    svgd_alpha = float(
        agent_val("alpha")
        or cfg.get('alpha')
        or kernel_alpha
        or 10.0
    )
    print(
        f"Using REINFORCE update. Number of agents: {M} with learning_rate_svgd: {learning_rate_svgd}, "
        f"λ: {lambda_}, svgd_alpha: {svgd_alpha}, advantage={advantage_cfg}, kernel={kernel_name}"
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    N = dim

    write_logs = bool(cfg.get('write_logs', False))
    pathResult = None

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
        learning_rate_svgd=learning_rate_svgd,
        enable_visualization=visualization_enabled,
        svgd_alpha=svgd_alpha,
        advantage_cfg=advantage_cfg,
        kernel_config=kernel_cfg,
    ).to(device)
    name_file_result = None
    if (type_problem == "QUBO"):
        list_scores = get_Score_trajectoriesQUBO_cuda(strategy, N, nb_instances_test, nb_restarts, budget, lambda_, tensor_Q_test, device, verbose, name_file_result, enable_visualization=visualization_enabled)

    elif (type_problem == "NK" or type_problem == "NK3"):
        list_scores = get_Score_trajectoriesNK_cuda(strategy, N,  type_instance, D, nb_instances_test, nb_restarts, 
                                                    budget, lambda_,
                                                    vectorIndex_th, tensor_matrix_locus,
                                                    tensor_matrix_contrib, device, verbose, name_file_result)
        
    print(list_scores)
    average_test_score = np.mean(list_scores)

    print("average_test_score : " + str(average_test_score))

if __name__ == '__main__':
    # Run hydra main
    main()
