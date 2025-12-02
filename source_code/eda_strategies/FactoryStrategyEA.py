
import torch.nn as nn
import torch
from eda_strategies.UMDA import UMDA
from eda_strategies.PBIL import PBIL
from eda_strategies.PPO_EDA import PPO_EDA
from eda_strategies.MultiAgentUnivariate.MultiAgentUnivariateEDA import MultiAgentUnivariateEDA 



class FactoryStrategyEA:

    def createStrategyEA(
        self,
        typeStrategy,
        N,
        lambda_,
        device,
        dim_variables,
        M,
        learning_rate,
        learning_rate_svgd,
        enable_visualization=False,
        svgd_rho=10.0,
    ):
        match typeStrategy:
            case "UMDA":
                return UMDA(N, lambda_, device)
            case "PBIL":
                return PBIL(N, lambda_, device)
            case "PPO-EDA":
                return MultiAgentUnivariateEDA(
                    N,
                    lambda_,
                    dim_variables,
                    M,
                    device=device,
                    learning_rate=learning_rate,
                    learning_rate_svgd=learning_rate_svgd,
                    enable_visualization=enable_visualization,
                    svgd_rho=svgd_rho,
                )
            case _:
                raise ValueError(f"Unknown strategy type: {typeStrategy}")
