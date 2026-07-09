
import torch.nn as nn
import torch
from eda_strategies.UMDA import UMDA
from eda_strategies.PBIL import PBIL
from eda_strategies.PPO_EDA import PPO_EDA
from eda_strategies.MultiAgentUnivariate.SVGD_EDA import SVGD_EDA 



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
        decay_enabled=True,
        advantage_cfg=None,
        kernel_config=None,
        no_interact=False,
        no_repulsion=False,
        is_nk3=False,
        ppo_active=False,
        ppo_epochs=4,
        ppo_mode='clip',
        clip_eps=0.2,
        kl_beta=1.0,
        kl_target_kl=None,
        kl_beta_max=100.0,
        kl_beta_min=1e-4,
        trpo_kl_threshold=0.01,
        trpo_backoff_max_tries=4,
    ):
        match typeStrategy:
            case "UMDA":
                return UMDA(N, lambda_, device)
            case "PBIL":
                return PBIL(N, lambda_, device)
            case "PPO-EDA":
                return SVGD_EDA(
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
                    decay_enabled=decay_enabled,
                    advantage_cfg=advantage_cfg,
                    kernel_config=kernel_config,
                    no_interact=no_interact,
                    no_repulsion=no_repulsion,
                    is_nk3=is_nk3,
                    ppo_active=ppo_active,
                    ppo_epochs=ppo_epochs,
                    ppo_mode=ppo_mode,
                    clip_eps=clip_eps,
                    kl_beta=kl_beta,
                    kl_target_kl=kl_target_kl,
                    kl_beta_max=kl_beta_max,
                    kl_beta_min=kl_beta_min,
                    trpo_kl_threshold=trpo_kl_threshold,
                    trpo_backoff_max_tries=trpo_backoff_max_tries,
                )
            case _:
                raise ValueError(f"Unknown strategy type: {typeStrategy}")
