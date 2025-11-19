import torch
import torch.nn as nn
from eda_strategies.Abstract_EDA import Abstract_EDA


class UnivariateBase(Abstract_EDA, nn.Module):
    """
    Version univariée propre : chaque variable Xi est indépendante,
    p(Xi,j=1) = sigmoid(theta_i_j) avec theta_i_j appris via une couche linéaire.
    i = instance, j = variable
    """

    def __init__(self, N, lambda_, typeModel, dim_variables, learning_rate, device):
        Abstract_EDA.__init__(self, N, lambda_, device)
        nn.Module.__init__(self)

        self.typeModel = typeModel
        self.learning_rate = learning_rate
        self.dim_variables = dim_variables
        self.lambda_ = lambda_
        self.N = N  # nombre de variables
        self.device = device
        self.theta = None
        self.nb_instances = 0
        self.register_buffer("baseline", torch.empty(0, dtype=torch.float32), persistent=False)
        self.last_theta_grad = None

    def forward(self):
        """
        Retourne les probabilités p(x_i=1) = sigmoid(theta_i)
        theta_i sont les poids de la couche linéaire.
        """
        probs = torch.sigmoid(self.theta)
        return probs  # (B, N)

    def reset_learned_parameters(self, nb_instances):
        """Réinitialise les paramètres et sauvegarde le nombre d'instances."""
        self.theta = nn.Parameter(torch.zeros((nb_instances, self.N), device=self.device))
        self.nb_instances = nb_instances  # B
        # Agent-specific state (optimizers, beta_vector, etc.) is initialized
        # inside the concrete agent classes (PPOAgent / REINFORCEAgent).

    def sample_solutions(self):
        probs = torch.sigmoid(self.theta)  # (B, N)
        probs = torch.clamp(torch.nan_to_num(probs, nan=0.5), 1e-6, 1 - 1e-6)
        u = torch.rand((self.nb_instances, self.lambda_, self.N), device=self.device)
        samples = (u < probs.unsqueeze(1)).float().unsqueeze(-1)  # (B, λ, N, 1)
        return samples

    def kl_divergence(self, p, q):
        eps = 1e-7
        p = torch.clamp(torch.nan_to_num(p, nan=0.5), eps, 1 - eps)
        q = torch.clamp(torch.nan_to_num(q, nan=0.5), eps, 1 - eps)
        return p * (torch.log(p) - torch.log(q)) + (1 - p) * (torch.log(1 - p) - torch.log(1 - q))


class PPOAgent(UnivariateBase):
    """Agent qui utilise PPO pour mettre à jour la distribution univariée."""

    def __init__(self, N, lambda_, beta, typeModel, dim_variables, learning_rate, device, agent_number, K_steps, beta_adapt, delta_target):
        super().__init__(N, lambda_, typeModel, dim_variables, learning_rate, device)
        self.beta = beta
        self.agent_number = agent_number
        self.K_steps = K_steps
        self.beta_adapt = beta_adapt
        self.delta_target = delta_target
        self.opt_ppo = None
        self.register_buffer("beta_vector", torch.empty(0, dtype=torch.float32), persistent=False)

    def reset_learned_parameters(self, nb_instances):
        # initialize common params in base
        super().reset_learned_parameters(nb_instances)
        # beta vector and baseline used by PPO
        self.beta_vector.resize_(nb_instances).fill_(1.0)
        self.baseline.resize_(nb_instances).zero_()
        # PPO optimizer
        self.opt_ppo = torch.optim.Adam([self.theta], lr=self.learning_rate)

    def updateDistribution(self, solutionList, scoreList):
        B = self.nb_instances
        N = self.N  # nombre de variables
        λ = self.lambda_
        device = self.device
        scoreList = scoreList  # (B, λ)
        K_steps = self.K_steps
        total_loss = 0.0
        with torch.no_grad():
            # capture de πθk
            p_old = torch.sigmoid(self.theta).clamp(1e-10, 1 - 1e-10)  # (B, N)
        # Données batchées
        Dk = solutionList.squeeze(-1)  # (B, λ, N) => solutionList est (B, λ, N, 1)
        fitnesses = scoreList
        if self.baseline is None:
            self.baseline = torch.zeros(B, device=device)  # (B,)
        adv = fitnesses - self.baseline.unsqueeze(1)  # (B, λ)
        adv = (adv - adv.mean(dim=1, keepdim=True)) / (adv.std(dim=1, keepdim=True) + 1e-10)
        # log πθk(a|s)
        p_old_exp = p_old.unsqueeze(1).expand(-1, λ, -1)  # (B, λ, N) => pour comparer avec Dk
        log_pi_old = torch.where(
            Dk == 1.0,
            torch.log(p_old_exp + 1e-10),
            torch.log(1.0 - p_old_exp + 1e-10)  # (B, λ, N)
        ).sum(dim=2)  # (B, λ)
        for _ in range(K_steps):
            # πθ(a|s)
            p_new = torch.sigmoid(self.theta).clamp(1e-10, 1 - 1e-10)  # (B, N)
            p_new_exp = p_new.unsqueeze(1).expand(-1, λ, -1)  # (B, λ, N)
            # log πθ(a|s) et ratio
            log_pi_new = torch.where(
                Dk == 1.0,
                torch.log(p_new_exp + 1e-10),
                torch.log(1.0 - p_new_exp + 1e-10)
            ).sum(dim=2)  # (B, λ)
            ratio = torch.exp(log_pi_new - log_pi_old)  # (B, λ)
            L_per_i = (ratio * adv).mean(dim=1)  # (B,)
            KL_mean_i = self.kl_divergence(p_old, p_new).mean(dim=1)  # (B,)
            if self.beta_adapt is True:
                loss_i = -L_per_i + self.beta_vector * KL_mean_i  # (B,)
            else:
                loss_i = -L_per_i + self.beta * KL_mean_i  # (B,)
            loss = loss_i.sum()
            self.opt_ppo.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.theta], max_norm=1.0)
            self.opt_ppo.step()
            total_loss += loss.item()
        if self.theta.grad is None:
            self.last_theta_grad = torch.zeros_like(self.theta)
        else:
            self.last_theta_grad = self.theta.grad.detach().clone()
        with torch.no_grad():
            p_new_final = torch.sigmoid(self.theta).clamp(1e-10, 1 - 1e-10)
        if self.beta_adapt is True:
            with torch.no_grad():
                KL_final_i = self.kl_divergence(p_old, p_new_final).mean(dim=1)  # (B,)
            mask_up = KL_final_i >= 1.5 * self.delta_target
            mask_down = KL_final_i <= (self.delta_target / 1.5)
            self.beta_vector[mask_up] *= 2.0
            self.beta_vector[mask_down] *= 0.5
            self.beta_vector = torch.clamp(self.beta_vector, 1e-4, 10)
        return total_loss / self.nb_instances


class REINFORCEAgent(UnivariateBase):
    """Agent qui utilise REINFORCE pour mettre à jour la distribution univariée."""

    def __init__(self, N, lambda_, typeModel, dim_variables, learning_rate, device, agent_number):
        super().__init__(N, lambda_, typeModel, dim_variables, learning_rate, device)
        self.agent_number = agent_number
        self.opt_reinforce = None

    def reset_learned_parameters(self, nb_instances):
        super().reset_learned_parameters(nb_instances)
        # baseline used by reinforce
        self.baseline.resize_(nb_instances).zero_()
        # REINFORCE optimizer
        self.opt_reinforce = torch.optim.SGD([self.theta], lr=self.learning_rate)

    def updateDistribution(self, solutionList, scoreList):
        device = self.device
        scoreList = scoreList
        actions = solutionList.squeeze(-1)  # (nb_instances, λ, N)
        if self.baseline is None:
            self.baseline = torch.zeros(self.nb_instances, device=device)  # (nb_instances,)
        all_Pi_Theta = self.forward()  # (nb_instances, N)
        all_Pi_Theta_expanded = all_Pi_Theta.unsqueeze(1).expand(-1, self.lambda_, -1)  # (nb_instances, λ, N)
        fitness = scoreList  # (nb_instances, λ)
        Pi_selected = torch.where(actions == 1.0, all_Pi_Theta_expanded, 1.0 - all_Pi_Theta_expanded)  # (nb_instances, λ, N)
        log_Pi = torch.log(Pi_selected + 1e-10).sum(dim=2)  # (nb_instances, λ)
        advantages = (fitness - self.baseline.unsqueeze(1))  # (nb_instances, λ)
        loss_per_instance = torch.mean(advantages * log_Pi, dim=1)  # (nb_instances,)
        loss = -loss_per_instance.sum()
        self.opt_reinforce.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_reinforce.step()
        if self.theta.grad is None:
            self.last_theta_grad = torch.zeros_like(self.theta)
        else:
            self.last_theta_grad = self.theta.grad.detach().clone()
        with torch.no_grad():
            self.baseline = fitness.mean(dim=1)  # (nb_instances,)
        return loss_per_instance.mean()
