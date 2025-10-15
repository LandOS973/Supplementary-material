import torch
import torch.nn as nn
from eda_strategies.Abstract_EDA import Abstract_EDA
import random


class UnivariatePPOEDA(Abstract_EDA, nn.Module):
    """
    Version univariée propre : chaque variable Xi est indépendante, 
    p(Xi,j=1) = sigmoid(theta_i_j) avec theta_i_j appris via une couche linéaire.
    i = instance, j = variable
    """

    def __init__(self, N, lambda_, beta, typeModel, dim_variables, learning_rate=0.01, device="cuda:0", agent_number=0):
        Abstract_EDA.__init__(self, N, lambda_, device)
        nn.Module.__init__(self)

        self.typeModel = typeModel
        self.learning_rate = learning_rate
        self.dim_variables = dim_variables
        self.lambda_ = lambda_
        self.N = N
        self.device = device
        self.beta = beta
        self.theta = None
        self.logits = None
        self.optimizerG = None
        self.agent_number = agent_number 
        self.theta_old = None


    def forward(self):
        """
        Retourne les probabilités p(x_i=1) = sigmoid(theta_i)
        theta_i sont les poids de la couche linéaire.
        """
        probs = torch.sigmoid(self.theta)
        return probs  # (N,)

    def reset_learned_parameters(self, nb_instances):
        """Réinitialise les paramètres et sauvegarde le nombre d'instances."""
        self.theta = nn.Parameter(0.5 * torch.randn(nb_instances, self.N, device=self.device))
        self.theta_old = self.theta.clone().detach()
        self.nb_instances = nb_instances
        self.optimizerG = torch.optim.Adam([self.theta], lr=self.learning_rate)

    def sample_solutions(self):
        """
        Échantillonne λ solutions selon Bernoulli(p).
        Sortie : (nb_instances, λ, N, 1)
        """
        probs = self.forward() # (nb_instances, N)
        probs_pop = probs.unsqueeze(1).expand(-1, self.lambda_, -1)        # (nb_instances, λ, N)

        with torch.no_grad():
            samples = torch.bernoulli(probs_pop).unsqueeze(-1)             # (nb_instances, λ, N, 1)

        return samples.to(self.device)

    def updateDistribution(self, solutionList, scoreList):
        device = self.device
        eps_clip = 0.2  # clipping threshold
        solutionList = solutionList.to(device)
        scoreList = scoreList.to(device)

        solutions = solutionList.squeeze(-1)  # (nb_instances, λ, N)

        _, indices = torch.sort(scoreList, dim=1, descending=True)
        indices = indices.long()
        sorted_solutions = torch.gather(
            solutions,
            1,
            indices.unsqueeze(-1).expand(-1, -1, self.N)
        )

        n_elite = max(1, self.lambda_ // 3)
        elite = sorted_solutions[:, :n_elite, :].to(device)

        target_probs = elite.mean(dim=1).clamp(1e-6, 1 - 1e-6).detach()

        probs = torch.sigmoid(self.theta).clamp(1e-6, 1 - 1e-6)
        with torch.no_grad():
            old_probs = torch.sigmoid(self.theta_old).clamp(1e-6, 1 - 1e-6)

        advantage = (target_probs - probs).detach()
        ratio = (probs * (1 - old_probs)) / (old_probs * (1 - probs))
        ratio = ratio.clamp(0.1, 10.0)

        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
        ppo_objective = torch.min(unclipped, clipped)

        loss = -ppo_objective.mean()

        self.optimizerG.zero_grad()
        loss.backward()
        self.optimizerG.step()
        
        with torch.no_grad():
            self.theta_old = self.theta.clone().detach()

        return loss.item()



    def toString(self):
        return "Strategy_Univariate_PPO_EDA number " + str(self.agent_number)

class MultiAgentUnivariateEDA(Abstract_EDA, nn.Module):
    """
    Multi-agent collaboratif :
    - M agents travaillent sur toutes les instances
    - Budget λ divisé entre agents (λ/M solutions par agent)
    - Diversité via learning rates différents
    """

    def __init__(self, N, lambda_, beta, typeModel, dim_variables, M=4, device="cuda:0"):
        Abstract_EDA.__init__(self, N, lambda_, device)
        nn.Module.__init__(self)

        self.M = M
        self.N = N
        self.lambda_ = lambda_
        self.device = device

        self.lambda_per_agent = lambda_ // M
        remainder_lambda = lambda_ % M

        learning_rates = [0.02 + 0.02 * i for i in range(M)]

        self.agents = nn.ModuleList()
        self.agent_lambdas = []
        bonus_indices = random.sample(range(M), remainder_lambda) if remainder_lambda > 0 else []
        
        for i in range(M):
            agent_lambda = self.lambda_per_agent + (1 if i in bonus_indices else 0)
            self.agent_lambdas.append(agent_lambda)
            
            agent = UnivariatePPOEDA(
                N, 
                agent_lambda, 
                beta, 
                typeModel, 
                dim_variables, 
                learning_rates[i],
                device, 
                agent_number=i
            )
            self.agents.append(agent)

    def reset_learned_parameters(self, nb_instances):
        self.nb_instances = nb_instances
        print(f"Multi-agent collaboratif : {self.M} agents × {nb_instances} instances (λ total={self.lambda_})")
        
        for i, agent in enumerate(self.agents):
            agent.reset_learned_parameters(nb_instances)  
            print(f"  Agent {i}: λ={self.agent_lambdas[i]}, lr={agent.learning_rate:.4f}")

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
            agent_scores = scoreList[:, start_lambda:end_lambda]              # (nb_instances, λ_agent)
            loss = agent.updateDistribution(agent_solutions, agent_scores)
            total_loss += loss

            start_lambda = end_lambda

        return total_loss / self.M

    def forward(self):
        return torch.stack([agent.forward() for agent in self.agents], dim=0)

    def toString(self):
        return f"MultiAgent_Collaborative_M{self.M}_lambda{self.lambda_}"