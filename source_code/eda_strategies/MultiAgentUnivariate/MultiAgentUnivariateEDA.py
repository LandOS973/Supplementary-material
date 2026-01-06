import torch
import torch.nn as nn
from types import SimpleNamespace

from eda_strategies.Abstract_EDA import Abstract_EDA
from eda_strategies.MultiAgentUnivariate.SVGD.SVGD import SVGD
from eda_strategies.MultiAgentUnivariate.SVGD.kernels.rbf import RBF
from eda_strategies.MultiAgentUnivariate.SVGD.kernels.ppk import PPK
from eda_strategies.MultiAgentUnivariate.SVGD.kernels.PK import ProbabilityKernel
from eda_strategies.MultiAgentUnivariate.SVGD.kernels.HK import HammingKernel
from eda_strategies.MultiAgentUnivariate.advantage import AdvantageFactory


class MultiAgentUnivariateEDA(Abstract_EDA, nn.Module):
    """
    Multi-agent collaboratif :
    - Budget λ défini par agent (M * λ solutions au total)
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
        sigma=None,
        svgd_alpha=10.0,
        advantage_cfg=None,
        kernel_config=None,
    ):
        self.M = M
        self.N = N
        # λ is now defined per agent; total population is M * λ
        self.lambda_per_agent = int(lambda_)
        self.total_lambda = self.lambda_per_agent * self.M
        Abstract_EDA.__init__(self, N, self.total_lambda, device)
        nn.Module.__init__(self)

        # Keep legacy attribute name for downstream code expecting total λ
        self.lambda_ = self.total_lambda
        self.device = device
        self.learning_rate = learning_rate
        self.learning_rate_svgd = learning_rate_svgd
        self.enable_visualization = bool(enable_visualization)
        self.dim_variables = dim_variables
        self.svgd_alpha = float(svgd_alpha)
        self.advantage_strategy = AdvantageFactory.from_config(advantage_cfg)
        self.kernel_config = kernel_config or {}
        self.kernel_name = str(self.kernel_config.get("name", "hk")).lower()
        self.kernel_params = {}

        # expose agent-level info for monitoring code (hamming/KL, leaderboard)
        self.agent_lambdas = [self.lambda_per_agent for _ in range(self.M)]
        self.agents = []

        # interaction SVGD 
        kernel_impl = self._build_svgd_kernel(self.kernel_name, self.kernel_params)
        self.svgd = SVGD(kernel_impl, alpha=self.svgd_alpha)
        self.theta_history = []
        self.kernel_metric_history = []

        # Paramètres appris : theta (nb_instances, M, N) initialisé dans reset
        self.theta = None
        self.nb_instances = 0
        self.latest_advantages = None

        # Buffers (B, M)
        self.register_buffer("baseline", torch.empty(0, dtype=torch.float32), persistent=False)
        self.last_theta_grad = None

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
        init_sigma =0.1
        init_theta = torch.randn((nb_instances, self.M, self.N), device=self.device) * init_sigma

        # init_theta = torch.zeros((nb_instances, self.M, self.N), device=self.device)
        self.theta = nn.Parameter(init_theta)
        self._refresh_agent_views()

        # Baseline en version (B, M)
        self.baseline.resize_(nb_instances, self.M).zero_()

        # Historique de visualisation
        self.theta_history = []
        self.kernel_metric_history = []
        self.last_theta_grad = None
        if self.enable_visualization:
            self._record_theta()
        self.latest_advantages = None

    def sample_solutions(self):
        """
        Génère (nb_instances, λ, N, 1) en une seule fois.

        Chaque agent possède son propre budget λ_agent (= lambda_per_agent),
        on échantillonne (B, M, λ_agent, N), puis on aplati en (B, λ_total, N, 1)
        avec λ_total = M * λ_agent.
        """
        B, M, N = self.nb_instances, self.M, self.N
        λa = self.lambda_per_agent
        λ_total = self.total_lambda

        probs = self.forward()  # (B, M, N)
        probs = torch.clamp(torch.nan_to_num(probs, nan=0.5), 1e-10, 1 - 1e-10)

        # u : (B, M, λa, N)
        u = torch.rand((B, M, λa, N), device=self.device)
        samples_agents = (u < probs.unsqueeze(2)).float()  # (B, M, λa, N)

        # On concatène tous les agents sur la dimension λ : (B, λ, N, 1)
        # on tire tout les lamdba en une fois pour eviter les boucles
        samples = samples_agents.view(B, λ_total, N).unsqueeze(-1)
        return samples

    def updateDistribution(self, solutionList, scoreList):
        """
        Applique la mise à jour REINFORCE suivie de SVGD entre agents (si activé).
        """
        # RL update (vectorisé sur (B, M))
        total_loss = self._updateDistribution_REINFORCE(solutionList, scoreList)
        # SVGD entre agents 
        self._apply_svgd()
        if self.enable_visualization:
            self._record_theta()
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
        advantages = self.advantage_strategy.compute(
            fitness=fitness,
            baseline=baseline,
            theta=theta,
            actions=actions,
            probs=all_Pi_Theta_expanded,
            nb_instances=B,
            num_agents=M,
        )  # (BM, λa)

        loss_per_instance = torch.mean(advantages * log_Pi, dim=1)  # (BM,)
        loss = loss_per_instance.sum()
        with torch.no_grad():
            reshaped_adv = advantages.detach().view(B, M, λa)
            per_instance = reshaped_adv.view(B, self.lambda_)
            self.latest_advantages = per_instance.cpu()

        grad_theta, = torch.autograd.grad(loss, theta, create_graph=False)
        self.last_theta_grad = grad_theta.detach().clone().view(B, M, N)

        with torch.no_grad():
            baseline_new = fitness.mean(dim=1)  # (BM,)
            self.baseline = baseline_new.view(B, M)

        # moyenne sur tous les (B, M) comme avant (Moyenne sur B, puis sur M)
        return loss_per_instance.mean()

    def get_latest_advantages(self):
        if self.latest_advantages is None:
            return None
        return self.latest_advantages.detach().cpu()

    # =======================
    #   SVGD 
    # =======================

    def toString(self):
        return f"MultiAgent_Collaborative_M{self.M}_lambdaPerAgent{self.lambda_per_agent}"

    def _apply_svgd(self):
        """
        Applique un pas SVGD instance par instance en se basant sur les directions RL observées.
        Utilise self.last_theta_grad comme direction RL : (B, M, N)
        """
        if self.last_theta_grad is None:
            return

        theta = self.theta  # (B, M, N)
        score = self.last_theta_grad.detach()  # pas de rétroprop vers les agents

        with torch.enable_grad():
            phi = self.svgd.phi(theta, score)  # (B, M, N)
            kernel_stats = self.svgd.get_last_kernel_stats()
            if kernel_stats:
                self.kernel_metric_history.append(kernel_stats)

        with torch.no_grad():
            self.theta += self.learning_rate_svgd * phi

    # =======================
    #   Visualisation
    # =======================

    def _record_theta(self):
        if self.theta is None or self.nb_instances <= 0:
            return []
        with torch.no_grad():
            probs = torch.sigmoid(self.theta).detach() # (B, M, N)
        probs_final = [probs[:, m, :] for m in range(self.M)] # liste de (B, N) par agent
        if not probs_final:
            return
        self.theta_history.append(probs_final)
    
    def get_theta_history(self):
        return {"values": self.theta_history}

    def get_kernel_metric_history(self):
        return list(self.kernel_metric_history)

    def get_latest_kernel_metrics(self):
        if not self.kernel_metric_history:
            return None
        return self.kernel_metric_history[-1]

    def _refresh_agent_views(self):
        if self.theta is None:
            self.agents = []
            return
        self.agents = [SimpleNamespace(theta=self.theta[:, idx, :]) for idx in range(self.M)]

    def _build_svgd_kernel(self, kernel_name, kernel_params):
        kernel = kernel_name.lower()
        if kernel in ("hk", "hamming", "hammingkernel"):
            return HammingKernel()
        if kernel == "ppk":
            return PPK()
        if kernel == "rbf":
            gamma = self.kernel_config.get("gamma")
            return RBF(gamma=gamma if gamma is not None else 0.08)
        if kernel == "pk":
            gamma = self.kernel_config.get("gamma")
            return ProbabilityKernel(gamma=gamma if gamma is not None else 1.0)
        raise ValueError(f"Unsupported kernel '{kernel_name}'. Available kernels: hk, ppk, rbf, pk.")
