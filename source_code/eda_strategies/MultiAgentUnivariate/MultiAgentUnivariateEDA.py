import random

import torch
import torch.nn as nn

from eda_strategies.Abstract_EDA import Abstract_EDA
from eda_strategies.MultiAgentUnivariate.RL_agent import PPOAgent, REINFORCEAgent


class MultiAgentUnivariateEDA(Abstract_EDA, nn.Module):
    """
    Multi-agent collaboratif :
    - M agents travaillent sur toutes les instances
    - Budget λ divisé entre agents (λ/M solutions par agent)
    - Diversité via learning rates différents
    """

    def __init__(self, N, lambda_, beta, typeModel, dim_variables, M, device, updateMethod, K_steps, beta_adapt, delta_target, learning_rate):
        Abstract_EDA.__init__(self, N, lambda_, device)
        nn.Module.__init__(self)

        self.M = M
        self.N = N
        self.lambda_ = lambda_
        self.device = device

        self.lambda_per_agent = lambda_ // M
        remainder_lambda = lambda_ % M

        self.agents = nn.ModuleList()
        self.agent_lambdas = []
        bonus_indices = random.sample(range(M), remainder_lambda) if remainder_lambda > 0 else []

        for i in range(M):
            agent_lambda = self.lambda_per_agent + (1 if i in bonus_indices else 0)
            self.agent_lambdas.append(agent_lambda)

            # instantiate the appropriate agent class depending on updateMethod
            if isinstance(updateMethod, str) and updateMethod.upper() == "PPO":
                agent = PPOAgent(
                    N,
                    agent_lambda,
                    beta,
                    typeModel,
                    dim_variables,
                    learning_rate,
                    device,
                    agent_number=i,
                    K_steps=K_steps,
                    beta_adapt=beta_adapt,
                    delta_target=delta_target,
                ).to(device)
            else:
                agent = REINFORCEAgent(
                    N,
                    agent_lambda,
                    typeModel,
                    dim_variables,
                    learning_rate,
                    device,
                    agent_number=i,
                ).to(device)
            self.agents.append(agent)

    def reset_learned_parameters(self, nb_instances):
        self.nb_instances = nb_instances
        for agent in self.agents:
            agent.reset_learned_parameters(nb_instances)

    def sample_solutions(self):
        samples_list = []
        for agent in self.agents:
            samples = agent.sample_solutions()  # (nb_instances, λ_agent, N, 1)
            samples_list.append(samples)

        return torch.cat(samples_list, dim=1)  # (nb_instances, λ, N, 1)

    def updateDistribution(self, solutionList, scoreList):
        total_loss = 0.0
        start_lambda = 0

        for i, agent in enumerate(self.agents):
            agent_lambda = self.agent_lambdas[i]
            end_lambda = start_lambda + agent_lambda
            agent_solutions = solutionList[:, start_lambda:end_lambda, :, :]  # (nb_instances, λ_agent, N, 1)
            agent_scores = scoreList[:, start_lambda:end_lambda]  # (nb_instances, λ_agent)
            loss = agent.updateDistribution(agent_solutions, agent_scores)
            total_loss += loss

            start_lambda = end_lambda

        return total_loss / self.M

    def forward(self):
        return torch.stack([agent.forward() for agent in self.agents], dim=0)

    def toString(self):
        return f"MultiAgent_Collaborative_M{self.M}_lambda{self.lambda_}"
