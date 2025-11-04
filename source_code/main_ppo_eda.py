import numpy as np
import argparse
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


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Black-Box Combinatorial Optimization with Order-Invariant Reinforcement Learning')

    #General arguments
    parser.add_argument('type_problem', type=str, help='type_problem : QUBO, NK or NK3')
    parser.add_argument('dim', type=int, help='Instance size')
    parser.add_argument('type_instance', type=int,  help='Type instance. Corresponding to K for NK landscape, or to the type of PUBOi distribution for QUBO instances')
    
    # General options
    parser.add_argument('--type_strategy', type=str, default="PPO_EDA", help='type_strategy : PPO-EDA, UMDA, PBIL, ')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--device', type=str, default="cuda:0", help='device')
    parser.add_argument('--nb_instances_test', type=int, default=10, help="Number of different instances for the test")
    parser.add_argument('--nb_restarts', type=int, default=10, help="Nb restarts")
    parser.add_argument('--budget', type=int, default=10000, help='number of calls to the objective function')
    
    # Multivariate EDA variants
    parser.add_argument('--lambda_', type=int, default=10, help='lambda : size pop EDA')
    parser.add_argument('--typeModel', type=str, default="NeuralNet", help='typeModel')
    parser.add_argument('--isUnivariate', type=int, default=0, help='isUnivariate')
    parser.add_argument('--updateMethod', type=str, default="REINFORCE", help='updateMethod for univariate PPO-EDA')
    parser.add_argument('--numberHiddenLayersG', type=int, default=1, help='numberHiddenLayersG')
    parser.add_argument('--nh', type=int, default=20, help='nh')
    
    #RL options
    parser.add_argument('--beta', type=float, default=1, help='beta : KL coefficient')
    
    # RL variants
    parser.add_argument('--knownIG', action='store_true')
    parser.add_argument('--fixSamplingOrder', action='store_true')
    parser.add_argument('--fixUpdateOrder', action='store_true')
    parser.add_argument('--learnOrder', action='store_true')
    parser.add_argument('--dropoutGen', type=float, default=0.0, help='additive structural dropout during generation')
    parser.add_argument('--dropoutTrain', type=float, default=0.0, help='additive structural dropout during learning')
    parser.add_argument('--withoutCausalMaskTraining', action='store_true')

    # Univariate EDA
    parser.add_argument('--M', type=int, default=4, help='Number of independent univariate agents (only for univariate PPO-EDA)')


    args = parser.parse_args()

    device = args.device
    type_strategy = args.type_strategy
    dim = args.dim
    type_instance = args.type_instance
    type_problem = args.type_problem
    nb_restarts = args.nb_restarts
    nb_instances_test = args.nb_instances_test
    seed = args.seed
    lambda_ = args.lambda_
    verbose = args.verbose
    budget = args.budget
    typeModel = args.typeModel
    isUnivariate = args.isUnivariate
    knownIG = args.knownIG
    fixSamplingOrder = args.fixSamplingOrder
    fixUpdateOrder = args.fixUpdateOrder
    numberHiddenLayersG = args.numberHiddenLayersG
    nh = args.nh
    beta = args.beta
    dropoutGen = args.dropoutGen
    dropoutTrain = args.dropoutTrain
    withoutCausalMaskTraining = args.withoutCausalMaskTraining
    M = args.M
    updateMethod = args.updateMethod
    
    learnOrder = args.learnOrder

    typeStrategy = "PPO-EDA"
    
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    N = dim

    if not os.path.exists("results/results_Multivariate-RL-EDA/" + typeStrategy ):
        os.mkdir("results/results_Multivariate-RL-EDA/" + typeStrategy)

    if not os.path.exists("results/results_Multivariate-RL-EDA/" + typeStrategy + "/" + type_problem ):
        os.mkdir("results/results_Multivariate-RL-EDA/" + typeStrategy + "/" + str(type_problem))
        
    if not os.path.exists("results/results_Multivariate-RL-EDA/" + typeStrategy + "/" + type_problem + "/" + str(dim) ):
        os.mkdir("results/results_Multivariate-RL-EDA/" + typeStrategy + "/" + type_problem + "/" + str(dim))

    if not os.path.exists("results/results_Multivariate-RL-EDA/" + typeStrategy + "/" + type_problem + "/" + str(dim) + "/" + str(type_instance) ):
        os.mkdir("results/results_Multivariate-RL-EDA/" + typeStrategy + "/" + type_problem + "/" + str(dim) + "/" + str(type_instance))
    
    pathResult = "results/results_Multivariate-RL-EDA/" + typeStrategy + "/" + type_problem + "/" + str(dim) + "/" + str(type_instance) + "/"

    name_file_result = "Test_" + type_strategy + "_" + type_problem +  "_N_" +  str(N) + "_t_" +  str(type_instance) + "_lambda_"  + str(lambda_) + "_beta_"  + str(beta) + "_typeModel_" + str(typeModel) + "_learnOrder_" + str(learnOrder) + "_knownIG_" + str(knownIG) + "_fixSamplingOrder_" + str(fixSamplingOrder) + "_fixUpdateOrder_" + str(fixUpdateOrder) + "_L_" + str(numberHiddenLayersG) + "_nh_" + str(nh)  + "_dGen_" + str(dropoutGen) + "_dTrain_" + str(dropoutTrain) + "_wCMaskTrain_" + str(withoutCausalMaskTraining)   + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + str(seed) + ".txt"
    

    if (type_problem == "QUBO"):

        instance_path = "instances/QUBO/"
        tensor_Q_test = getTensorInstances_QUBO(instance_path, nb_instances_test, nb_restarts, N, type_instance, device,
                                                "test")
    elif(type_problem == "NK"):

        D = 2
        vectorIndex = np.zeros((type_instance + 1))
        for i in range(type_instance + 1):
            vectorIndex[i] = D ** (type_instance - i)
        vectorIndex_th = torch.tensor(vectorIndex, dtype=torch.float32).to(device)


        tensor_matrix_locus, tensor_matrix_contrib, tensor_Q_test = getTensorInstances_NK("instances/nk/" + str(dim) + "/" + str(type_instance) + "/", nb_instances_test, nb_restarts, lambda_, dim, D, type_instance, device)

    elif(type_problem == "NK3"):

        D = 3
        vectorIndex = np.zeros((type_instance + 1))
        for i in range(type_instance + 1):
            vectorIndex[i] = D ** (type_instance - i)
        vectorIndex_th = torch.tensor(vectorIndex, dtype=torch.float32).to(device)


        tensor_matrix_locus, tensor_matrix_contrib, tensor_Q_test = getTensorInstances_NK("instances/nk3/" + str(dim) + "/" + str(type_instance) + "/", nb_instances_test, nb_restarts, lambda_, dim, D, type_instance, device)



    factory = FactoryStrategyEA()


    if (type_problem == "NK3"):
        dim_variables = [3 for i in range(N)]
    else:
        dim_variables = None


    strategy = factory.createStrategyEA(typeStrategy, dim, lambda_, beta, device,  typeModel, numberHiddenLayersG, nh, isUnivariate, dropoutGen, dropoutTrain, withoutCausalMaskTraining, dim_variables, learnOrder, 1, M, updateMethod=updateMethod)
        
        
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
    
    


