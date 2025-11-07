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


# Replication code for the article "Black-Box Combinatorial Optimization with Order-Invariant Reinforcement Learning"


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):

    # Support keeping the original variable names used previously; read them from Hydra cfg
    device = cfg.device
    type_strategy = cfg.agent.name if 'agent' in cfg and 'name' in cfg.agent else cfg.get('type_strategy', "PPO_EDA")
    type_problem = cfg.problem.name if 'problem' in cfg and 'name' in cfg.problem else cfg.get('type_problem', 'QUBO')
    print(f"Running with problem type: {type_problem}")
    dim = cfg.problem.dim if 'problem' in cfg and 'dim' in cfg.problem else cfg.get('dim', 64)
    type_instance = cfg.problem.type_instance if 'problem' in cfg and 'type_instance' in cfg.problem else cfg.get('type_instance', 1)
    print(f"Running with dim={dim}, type_instance={type_instance}")
    nb_restarts = int(cfg.nb_restarts)
    nb_instances_test = int(cfg.nb_instances_test)
    seed = int(cfg.seed)
    lambda_ = int(cfg.get('lambda', cfg.get('lambda_', 10)))
    verbose = bool(cfg.verbose)
    budget = int(cfg.budget)
    typeModel = cfg.typeModel
    isUnivariate = int(cfg.isUnivariate)
    knownIG = bool(cfg.knownIG)
    fixSamplingOrder = bool(cfg.fixSamplingOrder)
    fixUpdateOrder = bool(cfg.fixUpdateOrder)
    numberHiddenLayersG = int(cfg.numberHiddenLayersG)
    nh = int(cfg.nh)
    beta = float(cfg.get('beta', 1.0))
    dropoutGen = float(cfg.dropoutGen)
    dropoutTrain = float(cfg.dropoutTrain)
    withoutCausalMaskTraining = bool(cfg.withoutCausalMaskTraining)
    M = int(cfg.agent.get('M', cfg.get('M', 1))) if 'agent' in cfg else int(cfg.get('M', 1))
    updateMethod = cfg.agent.get('updateMethod', cfg.get('updateMethod', 'REINFORCE')) if 'agent' in cfg else cfg.get('updateMethod', 'REINFORCE')
    K_steps = int(cfg.agent.get('K_steps', cfg.get('K_steps', 6))) if 'agent' in cfg else int(cfg.get('K_steps', 6))
    beta_adapt = bool(cfg.agent.get('Beta_adapt', cfg.get('Beta_adapt', False))) if 'agent' in cfg else bool(cfg.get('Beta_adapt', False))
    learnOrder = bool(cfg.learnOrder)
    delta_target = float(cfg.agent.get('delta_target', cfg.get('delta_target', 0.003))) if 'agent' in cfg else float(cfg.get('delta_target', 0.003))
    learning_rate = float(cfg.agent.get('learning_rate', cfg.get('learning_rate', 0.02))) if 'agent' in cfg else float(cfg.get('learning_rate', 0.02))

    typeStrategy = "PPO-EDA"

    print(f"Using update method: {updateMethod} Number of agents: {M} with learning_rate: {learning_rate} delta_target: {delta_target} , K_steps: {K_steps} beta_adapt: {beta_adapt}")
    if cfg.agent.updateMethod == "PPO":
        K_steps = cfg.agent.K_steps
        beta_adapt = cfg.agent.Beta_adapt
        delta_target = cfg.agent.delta_target
    else:
        # ignorés pour REINFORCE
        K_steps = 0
        beta_adapt = False
        delta_target = 0.0

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    N = dim

    # Build results path relative to this script file (so it works regardless of current working dir)
    script_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    pathResult = os.path.join(repo_root, "results", "results_Multivariate-RL-EDA", typeStrategy, str(type_problem), str(dim), str(type_instance)) + os.sep
    os.makedirs(pathResult, exist_ok=True)
    

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


    strategy = factory.createStrategyEA(typeStrategy, dim, lambda_, beta, device,  typeModel, numberHiddenLayersG, nh, isUnivariate, dropoutGen, dropoutTrain, withoutCausalMaskTraining, dim_variables, learnOrder, 1, M, updateMethod=updateMethod, K_steps=K_steps, beta_adapt=beta_adapt, delta_target=delta_target, learning_rate=learning_rate)
        
        
    if(knownIG):
        
        if(type_problem == "QUBO" or type_problem == "NK" or type_problem == "NK3"):
            DAG = tensor_Q_test.unsqueeze(1).repeat(1, lambda_, 1, 1).to(device)
            DAG = torch.where(DAG != 0, 1, 0)
        else:
            print("IG unknown")
            
        strategy.setKnownDAG(DAG)
    
    if(fixSamplingOrder):
        
        order = torch.tensor(np.arange(dim)).to(device)
        
        order = order.unsqueeze(0).unsqueeze(1)
        order = order.repeat(nb_instances_test*nb_restarts, lambda_, 1)
        strategy.setKnownOrder(order)
    
    if(fixUpdateOrder):
        strategy.setSameDagTraining()

    

    # Build result filename now that dim (and pathResult) are final
    name_file_result = "Test_" + type_strategy + "_" + type_problem +  "_N_" +  str(N) + "_t_" +  str(type_instance) + "_lambda_"  + str(lambda_) + "_beta_"  + str(beta) + "_typeModel_" + str(typeModel) + "_learnOrder_" + str(learnOrder) + "_knownIG_" + str(knownIG) + "_fixSamplingOrder_" + str(fixSamplingOrder) + "_fixUpdateOrder_" + str(fixUpdateOrder) + "_L_" + str(numberHiddenLayersG) + "_nh_" + str(nh)  + "_dGen_" + str(dropoutGen) + "_dTrain_" + str(dropoutTrain) + "_wCMaskTrain_" + str(withoutCausalMaskTraining)   + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + str(seed) + ".txt"

    if (type_problem == "QUBO"):
        list_scores = get_Score_trajectoriesQUBO_cuda(strategy, N, nb_instances_test, nb_restarts, budget, lambda_, tensor_Q_test, device, verbose, pathResult + name_file_result)
    
    elif (type_problem == "NK" or type_problem == "NK3"):
        list_scores = get_Score_trajectoriesNK_cuda(strategy, N,  type_instance, D, nb_instances_test, nb_restarts, 
                                                    budget, lambda_,
                                                    vectorIndex_th, tensor_matrix_locus,
                                                    tensor_matrix_contrib, device, verbose, pathResult + name_file_result)
        


    print(list_scores)
    average_test_score = np.mean(list_scores)

    print("average_test_score : " + str(average_test_score))
    


if __name__ == '__main__':
    # Run hydra main
    main()




