import torch
import torch.nn as nn

from eda_strategies.Abstract_EDA import Abstract_EDA
from eda_strategies.MultiAgentUnivariate.SVGD.SVGD import SVGD
from eda_strategies.MultiAgentUnivariate.SVGD.rbf import RBF


class MultiAgentUnivariateEDA(Abstract_EDA, nn.Module):
    """
    Multi-agent collaboratif :
    - Budget λ divisé entre agents (λ/M solutions par agent)
    - B => NOMBRE D'INSTANCES
    - M => NOMBRE D'AGENTS
    - N => NOMBRE DE VARIABLES
    """

    def __init__(
        self,
        N,
        lambda_,
        dim_variables,
        M,
        device,
        learning_rate,
        learning_rate_svgd=None,
        enable_visualization=False,
    ):
        Abstract_EDA.__init__(self, N, lambda_, device)
        nn.Module.__init__(self)

        # Hypothèse vectorisée : λ est divisible par M
        assert lambda_ % M == 0, "lambda_ % M != 0"

        self.M = M
        self.N = N
        self.lambda_ = lambda_
        self.device = device
        self.learning_rate = learning_rate
        self.learning_rate_svgd = learning_rate_svgd
        self.enable_visualization = bool(enable_visualization)
        self.dim_variables = dim_variables

        # λ par agent
        self.lambda_per_agent = lambda_ // M
        # expose agent-level info for monitoring code (hamming/KL, leaderboard)
        self.agent_lambdas = [self.lambda_per_agent for _ in range(self.M)]
        self.agents = [_AgentView(self, idx) for idx in range(self.M)]

        # interaction SVGD (simple constant pour l'instant)
        self.svgd = SVGD(RBF())
        self.theta_history = []
        self.last_final_snapshot = None

        # Paramètres appris : theta (nb_instances, M, N) initialisé dans reset
        self.theta = None
        self.nb_instances = 0

        # Buffers (B, M)
        self.register_buffer("baseline", torch.empty(0, dtype=torch.float32), persistent=False)

        # États d'optimisation globaux
        self.last_theta_grad = None
        self.optimizer = None

    def forward(self):
        """
        -theta (B, M, N) -> sigmoid -> probs (B, M, N) ] 0,1 [
        """
        if self.theta is None:
            raise RuntimeError("reset_learned_parameters doit être appelé avant forward().")
        return torch.sigmoid(self.theta)  # (B, M, N)

    def kl_divergence(self, p, q):
        eps = 1e-7
        p = torch.clamp(torch.nan_to_num(p, nan=0.5), eps, 1 - eps)
        q = torch.clamp(torch.nan_to_num(q, nan=0.5), eps, 1 - eps)
        return p * (torch.log(p) - torch.log(q)) + (1 - p) * (torch.log(1 - p) - torch.log(1 - q))

    def reset_learned_parameters(self, nb_instances):
        self.nb_instances = nb_instances

        # theta : (B, M, N)
        self.theta = nn.Parameter(
            torch.zeros((nb_instances, self.M, self.N), device=self.device)
        )

        # Baseline en version (B, M)
        self.baseline.resize_(nb_instances, self.M).zero_()

        # Optimizer REINFORCE
        self.optimizer = torch.optim.SGD([self.theta], lr=self.learning_rate)

        # Historique de visualisation
        self.theta_history = []
        self.last_final_snapshot = None
        self.last_theta_grad = None

    def sample_solutions(self):
        """
        Génère (nb_instances, λ, N, 1) en une seule fois.

        Chaque agent a λ/M solutions, on échantillonne (B, M, λ_agent, N),
        puis on aplati en (B, λ, N, 1) avec λ = M * λ_agent.
        """
        B, M, N = self.nb_instances, self.M, self.N
        λa = self.lambda_per_agent

        probs = self.forward()  # (B, M, N)
        probs = torch.clamp(torch.nan_to_num(probs, nan=0.5), 1e-10, 1 - 1e-10)

        # u : (B, M, λa, N)
        u = torch.rand((B, M, λa, N), device=self.device)
        samples_agents = (u < probs.unsqueeze(2)).float()  # (B, M, λa, N)

        # On concatène tous les agents sur la dimension λ : (B, λ, N, 1)
        # on tire tout les lamdba en une fois pour eviter les boucles
        samples = samples_agents.view(B, self.lambda_, N).unsqueeze(-1)
        return samples

    def updateDistribution(self, solutionList, scoreList):
        """
        Applique la mise à jour REINFORCE suivie de SVGD entre agents (si activé).
        """
        # RL update (vectorisé sur (B, M))
        total_loss = self._updateDistribution_REINFORCE(solutionList, scoreList)

        # Visualisation des proba avant / après SVGD (optionnel)
        if self.enable_visualization:
            rl_snapshot = self._capture_prob_snapshot()
            prev_snapshot = (
                self.last_final_snapshot if self.last_final_snapshot is not None else rl_snapshot
            )
        else:
            rl_snapshot = None
            prev_snapshot = None

        # SVGD entre agents 
        self._apply_svgd()

        if self.enable_visualization:
            final_snapshot = self._capture_prob_snapshot()
            self._record_theta_snapshot(prev_snapshot, rl_snapshot, final_snapshot)
            self.last_final_snapshot = final_snapshot

        return total_loss

    # =======================
    #   REINFORCE vectorisé
    # =======================

    def _updateDistribution_REINFORCE(self, solutionList, scoreList):
        B, M, N = self.nb_instances, self.M, self.N
        λa = self.lambda_per_agent
        BM = B * M
        actions = solutionList.view(BM, λa, N)
        fitness = scoreList.view(BM, λa)
        theta = self.theta.view(BM, N)
        baseline = self.baseline.view(BM)

        if self.baseline.numel() == 0:
            baseline = torch.zeros(BM, device=self.device)

        all_Pi_Theta = torch.sigmoid(theta)  # (BM, N)
        all_Pi_Theta = torch.clamp(all_Pi_Theta, 1e-6, 1 - 1e-6)
        all_Pi_Theta_expanded = all_Pi_Theta.unsqueeze(1).expand(-1, λa, -1)  # (BM, λa, N)

        Pi_selected = torch.where(
            actions == 1.0,
            all_Pi_Theta_expanded,
            1.0 - all_Pi_Theta_expanded,
        )  # (BM, λa, N)

        log_Pi = torch.log(Pi_selected + 1e-10).sum(dim=2)  # (BM, λa)
        advantages = fitness - baseline.unsqueeze(1)  # (BM, λa)

        loss_per_instance = torch.mean(advantages * log_Pi, dim=1)  # (BM,)
        loss = -loss_per_instance.sum()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        #self.optimizer.step()

        if self.theta.grad is None:
            self.last_theta_grad = torch.zeros_like(self.theta)
        else:
            self.last_theta_grad = self.theta.grad.detach().clone()

        with torch.no_grad():
            baseline_new = fitness.mean(dim=1)  # (BM,)
            self.baseline = baseline_new.view(B, M)

        # moyenne sur tous les (B, M) comme avant (Moyenne sur B, puis sur M)
        return loss_per_instance.mean()

    # =======================
    #   SVGD 
    # =======================

    def toString(self):
        return f"MultiAgent_Collaborative_M{self.M}_lambda{self.lambda_}"

    def _apply_svgd(self):
        """
        Applique un pas SVGD instance par instance en se basant sur les directions RL observées.
        Utilise self.last_theta_grad comme direction RL : (B, M, N)
        """
        if self.last_theta_grad is None:
            return

        with torch.no_grad():
            theta = self.theta.detach()                     # (B, M, N)
            score = -self.last_theta_grad.detach()          # (B, M, N) un gradient par instance, agent et variable
            φ = self.svgd.phi(theta, score)  # (B, M, N)
            self.theta.data += (self.learning_rate_svgd * φ)

    # =======================
    #   Visualisation
    # =======================

    def _capture_prob_snapshot(self):
        if self.theta is None or self.nb_instances <= 0:
            return []
        with torch.no_grad():
            probs = torch.sigmoid(self.theta).detach().cpu()  # (B, M, N)
        return [probs[:, m, :] for m in range(self.M)] # liste de (B, N) par agent

    def _record_theta_snapshot(self, prev_snapshot, rl_snapshot, final_snapshot):
        if not rl_snapshot or not final_snapshot or not prev_snapshot:
            return
        entry = {"prev": prev_snapshot, "rl": rl_snapshot, "final": final_snapshot}
        self.theta_history.append(entry)

    def get_theta_history(self):
        return {"values": self.theta_history}


class _AgentView:
    """Lightweight view to expose per-agent theta for existing monitoring utilities."""

    def __init__(self, parent, agent_idx: int):
        self._parent = parent
        self._idx = agent_idx

    @property
    def theta(self):
        # returns a view with shape (B, N)
        return self._parent.theta[:, self._idx, :]
