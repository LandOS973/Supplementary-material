import random

import torch
import torch.nn as nn

from eda_strategies.Abstract_EDA import Abstract_EDA
from eda_strategies.MultiAgentUnivariate.RL_agent import PPOAgent, REINFORCEAgent
from eda_strategies.MultiAgentUnivariate.SVGD.SVGD import SVGD
from eda_strategies.MultiAgentUnivariate.SVGD.rbf import RBF


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

        # interaction SVGD (simple constant pour l'instant)
        self.svgd = SVGD(RBF())
        self.svgd_step_size = 1
        self.theta_history = []
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
        self.theta_history = []

    def sample_solutions(self):
        samples_list = []
        for agent in self.agents:
            samples = agent.sample_solutions()  # (nb_instances, λ_agent, N, 1)
            samples_list.append(samples)

        return torch.cat(samples_list, dim=1)  # (nb_instances, λ, N, 1)

    def updateDistribution(self, solutionList, scoreList):
        total_loss = 0.0
        rl_directions = []

        solution_chunks = torch.split(solutionList, self.agent_lambdas, dim=1)
        score_chunks = torch.split(scoreList, self.agent_lambdas, dim=1)

        for agent, agent_solutions, agent_scores in zip(self.agents, solution_chunks, score_chunks):
            loss = agent.updateDistribution(agent_solutions, agent_scores)
            total_loss += loss

            if agent.last_theta_grad is None:
                rl_step = torch.zeros_like(agent.theta)
            else:
                rl_step = -agent.last_theta_grad  # gradient descent direction
            rl_directions.append(rl_step.detach())
        if self.M > 1:
            self._apply_svgd(rl_directions)

        self._record_theta_snapshot()

        return total_loss / self.M

    def forward(self):
        return torch.stack([agent.forward() for agent in self.agents], dim=0)

    def toString(self):
        return f"MultiAgent_Collaborative_M{self.M}_lambda{self.lambda_}"

    def _apply_svgd(self, rl_directions):
        """
        Applique un pas SVGD instance par instance en se basant sur les directions RL observées.
        rl_directions : liste de tenseurs (nb_instances, N)
        """
        if not rl_directions:
            return

        with torch.no_grad():
            theta_stack = torch.stack([agent.theta.detach() for agent in self.agents], dim=1)  # (B, M, N)
        score_stack = torch.stack(rl_directions, dim=1)  # (B, M, N)
        phi_buffer = self.svgd.phi(theta_stack, score_stack)

        with torch.no_grad():
            for idx, agent in enumerate(self.agents):
                agent.theta.add_(self.svgd_step_size * phi_buffer[:, idx, :])

    def _record_theta_snapshot(self):
        if not self.agents or self.nb_instances <= 0:
            return

        snapshot = []
        with torch.no_grad():
            for agent in self.agents:
                probs = torch.sigmoid(agent.theta).detach().cpu()
                snapshot.append(probs)
        self.theta_history.append(snapshot)

    def get_theta_history(self):
        return {"values": self.theta_history}
