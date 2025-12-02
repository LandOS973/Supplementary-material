import numpy as np
import argparse
import hydra
from omegaconf import DictConfig, OmegaConf
import datetime
import torch
import os
import random
from random import sample
from eda_strategies.FactoryStrategyEA import FactoryStrategyEA
from eda_strategies.PBIL import PBIL
from eda_strategies.UMDA import UMDA
from time import time
from utils.walsh_expansion import WalshExpansion
from tqdm import tqdm
from environment.qubo import getTensorInstances_QUBO, get_Score_trajectoriesQUBO_cuda
from environment.nk import getTensorInstances_NK, get_Score_trajectoriesNK_cuda


import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True, formatter={"float_kind": lambda x: f"{x:.6f}"})

# Replication code for the article "Black-Box Combinatorial Optimization with Order-Invariant Reinforcement Learning"


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
    # safe getter for possibly nested agent configs (cfg.agent can be a string when only the group name is set)
    def oget(path, default=None):
        try:
            val = OmegaConf.select(cfg, path)
            return val if val is not None else default
        except Exception:
            return default
    lambda_cfg = oget('agent.lambda', cfg.get('lambda', cfg.get('lambda_', None)))
    lambda_ = int(lambda_cfg) if lambda_cfg is not None else 10
    verbose = bool(cfg.verbose)
    budget = int(cfg.budget)
    visualization_enabled = bool(cfg.get('visualization', True))
    lr_svgd_cfg = oget('agent.learning_rate_svgd', cfg.get('learning_rate_svgd', None))
    learning_rate_svgd = float(lr_svgd_cfg) if lr_svgd_cfg is not None else 0.5
    rho_cfg = oget('agent.rho', cfg.get('rho', None))
    svgd_rho = float(rho_cfg) if rho_cfg is not None else 10.0

    M = int(oget('agent.M', oget('M', 1)))
    lr_cfg = oget('agent.learning_rate', oget('learning_rate', None))
    learning_rate = float(lr_cfg) if lr_cfg is not None else 0.0
    typeStrategy = "PPO-EDA"

    print(f"Using REINFORCE update. Number of agents: {M} with learning_rate: {learning_rate}, "
          f"learning_rate_svgd: {learning_rate_svgd}, λ: {lambda_}, svgd_rho: {svgd_rho}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    N = dim

    # Build results path relative to this script file (so it works regardless of current working dir)
    script_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
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
        svgd_rho=svgd_rho,
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
