import torch
import torch.nn as nn
from types import SimpleNamespace

from eda_strategies.Abstract_EDA import Abstract_EDA
from eda_strategies.MultiAgentUnivariate.SVGD.SVGD import SVGD
from eda_strategies.MultiAgentUnivariate.SVGD.kernels.rbf import RBF
from eda_strategies.MultiAgentUnivariate.SVGD.kernels.no_interact import NoInteractKernel
from eda_strategies.MultiAgentUnivariate.advantage import AdvantageFactory


class SVGD_EDA(Abstract_EDA, nn.Module):
    """
    Multi-agent collaboratif :
    - Budget λ défini par agent (M * λ solutions au total)
    - B => NOMBRE D'INSTANCES
    - M => NOMBRE D'AGENTS
    - N => NOMBRE DE VARIABLES

    Mise à jour : PPO avec K époques internes + SVGD intra-boucle.
    """

    def __init__(
        self,
        N,
        lambda_,
        dim_variables,
        M,
        device,
        epsilon_svgd=None,
        enable_visualization=False,
        no_interact=False,
        no_repulsion=False,
        svgd_gamma=10.0,
        decay_start_ratio=0.8,
        decay_min_factor=0.1,
        decay_enabled=False,
        advantage_cfg=None,
        kernel_config=None,
        is_nk3=False,
        # PPO hyperparameters
        ppo_active=False,      # True => PPO (K époques), False => REINFORCE pur
        ppo_epochs=1,          # K : nombre d'époques internes (ignoré si ppo_active=False)
        kl_beta=1.0,           # β : pénalité KL, fixe
    ):
        self.M = M
        self.N = N
        self.lambda_per_agent = int(lambda_)
        self.total_lambda = self.lambda_per_agent * self.M
        Abstract_EDA.__init__(self, N, self.total_lambda, device)
        nn.Module.__init__(self)

        self.lambda_ = self.total_lambda
        self.device = device
        self.epsilon_svgd = epsilon_svgd
        self.enable_visualization = bool(enable_visualization)
        self.no_interact = bool(no_interact)
        self.no_repulsion = bool(no_repulsion)
        self.dim_variables = dim_variables
        self.is_nk3 = bool(is_nk3)
        self.use_categorical = bool(self.is_nk3 or self.dim_variables is not None)
        self.max_dim = 3 if self.is_nk3 else None
        if self.dim_variables is not None:
            if len(self.dim_variables) != self.N:
                raise ValueError(
                    f"dim_variables length ({len(self.dim_variables)}) must match N={self.N}."
                )
            if len(set(self.dim_variables)) != 1:
                raise ValueError(
                    "SVGD_EDA ne supporte que des variables categorielles de meme cardinalite "
                    f"(dim_variables={self.dim_variables})."
                )
            self.max_dim = int(self.dim_variables[0])
            if self.max_dim < 2:
                raise ValueError(f"Invalid categorical max_dim: {self.max_dim}")
        self.svgd_gamma = float(svgd_gamma)
        self.decay_start_ratio = float(decay_start_ratio)
        self.decay_min_factor = float(decay_min_factor)
        self.decay_enabled = bool(decay_enabled)

        # PPO (mode KL, beta fixe)
        self.ppo_active = bool(ppo_active)
        self.ppo_epochs = int(ppo_epochs)
        self.kl_beta = float(kl_beta)

        kernel_config_local = kernel_config or {}
        self.advantage_strategy = AdvantageFactory.from_config(advantage_cfg)
        self.kernel_config = kernel_config_local
        self.kernel_name = str(self.kernel_config.get("name", "rbf")).lower()
        self.prob_eps_clamp = float(self.kernel_config.get("prob_eps_clamp", 1e-3))

        self.agent_lambdas = [self.lambda_per_agent for _ in range(self.M)]
        self.agents = []

        kernel_impl = self._build_svgd_kernel(self.kernel_name)
        self.svgd = SVGD(kernel_impl, gamma=self.svgd_gamma, no_repulsion=self.no_repulsion)
        self.theta_history = []
        self._last_kernel_stats = None

        self.theta = None
        self.nb_instances = 0
        self.probs = None

        self.register_buffer("baseline", torch.empty(0, dtype=torch.float32), persistent=False)
        self.last_theta_grad = None

    def forward(self):
        """
        -theta (B, M, N) -> sigmoid -> probs (B, M, N) ] 0,1 [
        -theta (B, M, N, D) -> softmax -> probs (B, M, N, D)
        """
        if self.theta is None:
            raise RuntimeError("reset_learned_parameters doit être appelé avant forward().")
        if self.use_categorical:
            probs = torch.softmax(self.theta, dim=-1)
            probs = torch.nan_to_num(probs, nan=1.0 / float(probs.size(-1)))
            probs = torch.clamp(probs, self.prob_eps_clamp, 1.0 - self.prob_eps_clamp)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            self.probs = probs
            return probs
        probs = torch.sigmoid(self.theta)
        self.probs = torch.clamp(torch.nan_to_num(probs, nan=0.5), self.prob_eps_clamp, 1 - self.prob_eps_clamp)
        return probs

    def reset_learned_parameters(self, nb_instances):
        self.nb_instances = nb_instances

        init_sigma = 0.1
        if self.use_categorical:
            max_dim = self.max_dim or 3
            init_theta = torch.randn(
                (nb_instances, self.M, self.N, max_dim), device=self.device
            ) * init_sigma
        else:
            init_theta = torch.randn((nb_instances, self.M, self.N), device=self.device) * init_sigma

        self.theta = nn.Parameter(init_theta)
        self._refresh_agent_views()

        self.baseline.resize_(nb_instances, self.M).zero_()

        self.theta_history = []
        self._last_kernel_stats = None
        self.last_theta_grad = None
        if self.enable_visualization:
            self._record_theta()
        self.probs = None

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

        self.probs = self.forward()

        if self.use_categorical:
            probs = self.probs
            D = probs.size(-1)
            flat = probs.reshape(-1, D)
            samples_flat = torch.multinomial(flat, num_samples=λa, replacement=True)
            samples_agents = samples_flat.view(B, M, N, λa).permute(0, 1, 3, 2)
            samples = samples_agents.reshape(B, λ_total, N).unsqueeze(-1).float()
            return samples

        u = torch.rand((B, M, λa, N), device=self.device)
        samples_agents = (u < self.probs.unsqueeze(2)).float()

        samples = samples_agents.view(B, λ_total, N).unsqueeze(-1)
        return samples

    def sample_greedy_agent_solutions(self):
        """
        Génère une solution déterministe par agent :
        - binaire : arrondi (p >= 0.5)
        - catégoriel : argmax.
        Retourne (B, M, N, 1).
        """
        probs = self.forward()
        if self.use_categorical:
            greedy = torch.argmax(probs, dim=-1)
            return greedy.unsqueeze(-1).float()
        greedy = (probs >= 0.5).float()
        return greedy.unsqueeze(-1)

    def updateDistribution(self, solutionList, scoreList):
        """Applique la mise à jour (REINFORCE ou PPO) suivie de SVGD entre agents."""
        if self.ppo_active:
            total_loss = self._updateDistribution_PPO(solutionList, scoreList)
        else:
            total_loss = self._updateDistribution_REINFORCE(solutionList, scoreList)
            self._apply_svgd()
        if self.enable_visualization:
            self._record_theta()
        return total_loss

    def _compute_kl(self, pi_old_full, pi_new_full):
        """
        KL(π_old || π_new) par (instance×agent, variable), sommée sur N puis sur BM
        (même échelle que `surrogate`, pour l'objectif PPO pénalisé par -beta*KL).
        binary:       pi (BM, N)      — probabilité de x=1
        categorical:  pi (BM, N, D)   — distribution sur les D catégories
        """
        if self.use_categorical:
            kl = (pi_old_full * torch.log(pi_old_full / pi_new_full)).sum(dim=-1)  # (BM, N)
        else:
            kl = pi_old_full * torch.log(pi_old_full / pi_new_full) + \
                (1.0 - pi_old_full) * torch.log((1.0 - pi_old_full) / (1.0 - pi_new_full))  # (BM, N)
        return kl.sum(dim=-1).sum()

    def _prepare_step(self, solutionList, scoreList):
        """
        Reshape (solutions, scores) en (BM, λa, N), calcule les avantages Ŝ
        (fixes, sans gradient) et met à jour la baseline. Partagé par REINFORCE
        et PPO, qui n'en diffèrent qu'après ce point.
        """
        B, M, N = self.nb_instances, self.M, self.N
        BM = B * M
        λa = self.lambda_per_agent

        indivduals = solutionList.view(BM, λa, N)
        fitness = scoreList.view(BM, λa)
        baseline = self.baseline.view(BM) if self.baseline.numel() > 0 else torch.zeros(BM, device=self.device)

        if self.use_categorical:
            D = self.probs.size(-1)
            theta_flat = self.theta.view(BM, N, D)
            probs_exp = self.probs.view(BM, N, D).unsqueeze(1).expand(-1, λa, -1, -1)
        else:
            theta_flat = self.theta.view(BM, N)
            probs_exp = self.probs.view(BM, N).unsqueeze(1).expand(-1, λa, -1)

        advantages = self.advantage_strategy.compute(
            fitness=fitness,
            baseline=baseline,
            theta=theta_flat,
            indivduals=indivduals,
            probs=probs_exp,
            nb_instances=B,
            num_agents=M,
        ).detach()

        with torch.no_grad():
            self.baseline = fitness.mean(dim=1).view(B, M)

        return indivduals, advantages

    def _updateDistribution_REINFORCE(self, solutionList, scoreList):
        """
        Mise à jour REINFORCE pure : ∇_θ E[A · log π_θ(x)].
        Identique au comportement de la branche main (ppo_active=False).
        _apply_svgd() est appelé par updateDistribution après ce retour.
        """
        indivduals, advantages = self._prepare_step(solutionList, scoreList)
        log_Pi = torch.log(self._compute_pi_per_pos(indivduals) + 1e-10).sum(dim=2)  # (BM, λa)

        loss = torch.mean(advantages * log_Pi, dim=1).sum()

        (grad_theta,) = torch.autograd.grad(loss, self.theta, create_graph=False, retain_graph=True)
        self.last_theta_grad = grad_theta.detach().clone()

        return loss

    def _compute_pi_per_pos(self, indivduals):
        """
        Retourne π_θ(x_n) (probabilité brute) pour chaque position n.
        indivduals : (BM, λa, N)
        Retourne   : (BM, λa, N)
        """
        BM, λa, N = indivduals.shape
        if self.use_categorical:
            D = self.probs.size(-1)
            probs = self.probs.view(BM, N, D)
            probs_exp = probs.unsqueeze(1).expand(-1, λa, -1, -1)       # (BM, λa, N, D)
            indices = indivduals.long().unsqueeze(-1)                    # (BM, λa, N, 1)
            return probs_exp.gather(-1, indices).squeeze(-1)             # (BM, λa, N)
        else:
            probs = self.probs.view(BM, N)
            probs_exp = probs.unsqueeze(1).expand(-1, λa, -1)           # (BM, λa, N)
            return torch.where(indivduals == 1.0, probs_exp, 1.0 - probs_exp)  # (BM, λa, N)

    def _updateDistribution_PPO(self, solutionList, scoreList):
        """
        Implémente SVGD-EDA-PPO (mode KL, beta fixe) :
          θ_old ← θ
          for k = 1..K :
              r_jl = exp(log π_θ - log π_θ_old)
              objective = E[r_jl · Ŝ] - beta · KL(π_old ‖ π_new)
              θ_i += ε · SVGD_phi(θ, ∇objective)
        """
        indivduals, advantages = self._prepare_step(solutionList, scoreList)
        BM, _, N = indivduals.shape

        # --- π_θ_old figées pour les K époques (ratio + KL) ---
        with torch.no_grad():
            pi_old_per_pos = self._compute_pi_per_pos(indivduals).detach()  # (BM, λa, N)
            pi_old_full = self.probs.view(BM, N, -1).detach() if self.use_categorical \
                else self.probs.view(BM, N).detach()

        # --- Boucle K époques PPO avec SVGD intra-boucle ---
        last_surrogate_mean = None
        adv_exp = advantages.unsqueeze(-1)  # (BM, λa, 1) — précalculé hors boucle

        for epoch in range(self.ppo_epochs):
            if epoch > 0:
                self.probs = self.forward()

            # Ratio par position : r_n = π_θ_new(x_n) / π_θ_old(x_n)
            pi_new_per_pos = self._compute_pi_per_pos(indivduals)  # (BM, λa, N)
            ratio = pi_new_per_pos / pi_old_per_pos                # (BM, λa, N)
            surrogate = (ratio * adv_exp).sum(dim=-1)              # (BM, λa) — pas de clipping

            pi_new_full = self.probs.view(BM, N, -1) if self.use_categorical \
                else self.probs.view(BM, N)
            kl_sum = self._compute_kl(pi_old_full, pi_new_full)
            objective = surrogate.mean(dim=1).sum() - self.kl_beta * kl_sum

            # retain_graph=False : le graph est reconstruit à chaque epoch via forward()
            (grad_theta,) = torch.autograd.grad(
                objective, self.theta, create_graph=False, retain_graph=False
            )
            self.last_theta_grad = grad_theta.detach().clone()

            # Pas SVGD : θ_i += ε · φ(θ)  — à l'intérieur de la boucle K
            self._apply_svgd()

            last_surrogate_mean = float(surrogate.detach().mean(dim=1).mean().item())

        val = last_surrogate_mean if last_surrogate_mean is not None else 0.0
        return torch.tensor(val, device=self.device)

    def _apply_svgd(self):
        """
        Applique un pas SVGD instance par instance en se basant sur les directions RL observées.
        Utilise self.last_theta_grad comme direction RL : (B, M, N)
        """
        if self.last_theta_grad is None:
            return

        with torch.no_grad():
            phi = self.svgd.phi(self.theta, self.last_theta_grad.detach())
            kernel_stats = self.svgd.get_last_kernel_stats()
            if kernel_stats:
                self._last_kernel_stats = kernel_stats
            self.theta += self.epsilon_svgd * phi
            self.probs = None

    def decay_svgd_gamma(self, current_iter: int, total_iters: int) -> None:
        if not self.decay_enabled or self.no_interact or self.decay_start_ratio >= 1.0 or self.decay_min_factor >= 1.0:
            return
        progress = (current_iter + 1) / float(total_iters)
        start = self.decay_start_ratio
        min_factor = self.decay_min_factor

        if progress < start:
            return
        else:
            t = (progress - start) / (1.0 - start)
            factor = 1.0 - t * (1.0 - min_factor)

        target_gamma = self.svgd_gamma * factor
        self.svgd.gamma = float(target_gamma)

    def _record_theta(self):
        if self.theta is None or self.nb_instances <= 0:
            return
        with torch.no_grad():
            probs = self.probs if self.probs is not None else self.forward()
        probs_final = [probs[:, m, :] for m in range(self.M)]
        if not probs_final:
            return
        self.theta_history.append(probs_final)

    def get_theta_history(self):
        return {"values": self.theta_history}

    def get_latest_kernel_metrics(self):
        return self._last_kernel_stats

    def get_latest_force_stats(self):
        return self.svgd.get_last_force_stats()

    def _refresh_agent_views(self):
        if self.theta is None:
            self.agents = []
            return
        self.agents = [SimpleNamespace(theta=self.theta[:, idx, :]) for idx in range(self.M)]

    def _build_svgd_kernel(self, kernel_name):
        kernel = kernel_name.lower()
        if self.no_interact or kernel in ("no_interact", "no-interact", "identity", "none"):
            return NoInteractKernel()
        if kernel == "rbf":
            return RBF()
        raise ValueError(
            f"Unsupported kernel '{kernel_name}'. Available kernels: rbf, no_interact."
        )
