
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
        epsilon_svgd,
        enable_visualization=False,
        svgd_gamma=10.0,
        decay_start_ratio=0.8,
        decay_min_factor=0.1,
        advantage_cfg=None,
        kernel_config=None,
        no_interact=False,
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
                    epsilon_svgd=epsilon_svgd,
                    enable_visualization=enable_visualization,
                    svgd_gamma=svgd_gamma,
                    decay_start_ratio=decay_start_ratio,
                    decay_min_factor=decay_min_factor,
                    advantage_cfg=advantage_cfg,
                    kernel_config=kernel_config,
                    no_interact=no_interact,
                )
            case _:
                raise ValueError(f"Unknown strategy type: {typeStrategy}")
