
import torch.nn as nn
import torch
from eda_strategies.UMDA import UMDA
from eda_strategies.PBIL import PBIL
from eda_strategies.PPO_EDA import PPO_EDA
from eda_strategies.MultiAgentUnivariateEDA import MultiAgentUnivariateEDA 



class FactoryStrategyEA:

    def createStrategyEA(self, typeStrategy, N, lambda_, beta, device, typeModel, numberHiddenLayersG, nh, isUnivariate, dropoutGen, dropoutTrain, withoutCausalMaskTraining, dim_variables, learnDAG, noise_rescale, M, updateMethod="REINFORCE", K_steps=6):
        print("Création de la stratégie : " + typeStrategy)
        if (typeStrategy == "UMDA"):
            return UMDA(N, lambda_, device)

        elif(typeStrategy == "PBIL"):

            return PBIL(N, lambda_, device)


        elif (typeStrategy == "PPO-EDA"):
            if(isUnivariate):
                print("Using update method: " + updateMethod)
                if(updateMethod == "PPO"):
                    print("Using K_steps: " + str(K_steps))
                return MultiAgentUnivariateEDA(N, lambda_, beta, typeModel, dim_variables, M, device=device, updateMethod=updateMethod, K_steps=K_steps)
            else:
                return PPO_EDA(N,  lambda_, beta, device, typeModel,numberHiddenLayersG, nh, isUnivariate, dropoutGen, dropoutTrain, withoutCausalMaskTraining, dim_variables, learnDAG, noise_rescale)
