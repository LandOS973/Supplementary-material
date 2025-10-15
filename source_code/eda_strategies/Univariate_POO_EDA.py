import torch
import torch.nn as nn
from eda_strategies.Abstract_EDA import Abstract_EDA


class UnivariatePPOEDA(Abstract_EDA, nn.Module):
    """
    Version univariée propre : chaque variable Xi est indépendante, 
    p(Xi=1) = sigmoid(theta_i) avec theta_i appris via une couche linéaire.
    """

    def __init__(self, N, lambda_, beta, typeModel, dim_variables, learning_rate=0.01, device="cuda:0"):
        Abstract_EDA.__init__(self, N, lambda_, device)
        nn.Module.__init__(self)

        self.typeModel = typeModel
        self.learning_rate = learning_rate
        self.dim_variables = dim_variables
        self.lambda_ = lambda_
        self.N = N
        self.device = device
        self.beta = beta

        # Couche linéaire : stocke un vecteur de paramètres θ
        self.linear = nn.Linear(N, N, bias=False)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.5)

        # Optimiseur
        self.optimizerG = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self):
        """
        Retourne les probabilités p(x_i=1) = sigmoid(theta_i)
        theta_i sont les poids de la couche linéaire.
        """
        logits = self.linear.weight[0]  # (N,)
        probs = torch.sigmoid(logits)
        print(probs)
        return probs  # (N,)

    def reset_learned_parameters(self, nb_instances):
        """Réinitialise les paramètres et sauvegarde le nombre d'instances."""
        self.nb_instances = nb_instances
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.5)
        self.optimizerG = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def sample_solutions(self):
        """
        Échantillonne λ solutions selon Bernoulli(p).
        Sortie : (nb_instances, λ, N, 1)
        """
        probs = self.forward().unsqueeze(0).expand(self.nb_instances, -1)  # (nb_instances, N)
        probs_pop = probs.unsqueeze(1).expand(-1, self.lambda_, -1)        # (nb_instances, λ, N)

        with torch.no_grad():
            samples = torch.bernoulli(probs_pop).unsqueeze(-1)             # (nb_instances, λ, N, 1)

        return samples.to(self.device)

    def updateDistribution(self, solutionList, scoreList):
        device = self.device

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

        n_elite = max(1, self.lambda_ // 5)
        elite = sorted_solutions[:, :n_elite, :].to(device)
        target_probs = elite.mean(dim=1).to(device)  # (nb_instances, N)

        probs = self.forward().unsqueeze(0).expand(self.nb_instances, -1).to(device)
        old_probs = probs.detach()

        # PPO-style loss + KL
        ppo_loss = torch.mean((probs - target_probs) ** 2)

        kl = torch.distributions.kl.kl_divergence(
            torch.distributions.Bernoulli(probs=probs),
            torch.distributions.Bernoulli(probs=old_probs)
        ).mean()

        loss = ppo_loss + self.beta * kl

        self.optimizerG.zero_grad()
        loss.backward()
        self.optimizerG.step()

        return loss.item()


    def toString(self):
        return "Strategy_Univariate_PPO_EDA"
