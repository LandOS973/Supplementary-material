import math
import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace

from eda_strategies.Abstract_EDA import Abstract_EDA
from eda_strategies.MultiAgentUnivariate.SVGD.SVGD import SVGD
from eda_strategies.MultiAgentUnivariate.SVGD.kernels.rbf import RBF
from eda_strategies.MultiAgentUnivariate.SVGD.kernels.ppk import PPK
from eda_strategies.MultiAgentUnivariate.SVGD.kernels.JSD import JSD
from eda_strategies.MultiAgentUnivariate.SVGD.kernels.PK import ProbabilityKernel
from eda_strategies.MultiAgentUnivariate.SVGD.kernels.HK import HammingKernel
from eda_strategies.MultiAgentUnivariate.SVGD.kernels.FR import FisherRaoKernel
from eda_strategies.MultiAgentUnivariate.SVGD.kernels.no_interact import NoInteractKernel
from eda_strategies.MultiAgentUnivariate.advantage import AdvantageFactory


class SVGD_EDA(Abstract_EDA, nn.Module):
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
        epsilon_svgd=None,
        enable_visualization=False,
        no_interact=False,
        no_repulsion=False,
        sigma=None,
        svgd_gamma=10.0,
        decay_start_ratio=0.8,
        decay_min_factor=0.1,
        decay_enabled=False,
        advantage_cfg=None,
        kernel_config=None,
        is_nk3=False,
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
            self.max_dim = int(max(self.dim_variables)) if self.dim_variables else None
            if self.max_dim is None or self.max_dim < 2:
                raise ValueError(f"Invalid categorical max_dim: {self.max_dim}")
            mask = torch.ones(self.N, self.max_dim)
            for idx, dim in enumerate(self.dim_variables):
                if dim < self.max_dim:
                    mask[idx, dim:] = 0.0
            self.register_buffer(
                "mask",
                mask.unsqueeze(0).unsqueeze(0),
                persistent=False,
            )
        else:
            self.mask = None
        self.svgd_gamma = float(svgd_gamma)
        self.decay_start_ratio = float(decay_start_ratio)
        self.decay_min_factor = float(decay_min_factor)
        self.decay_enabled = bool(decay_enabled)
        kernel_config_local = kernel_config or {}
        advantage_cfg_local = advantage_cfg
        if isinstance(advantage_cfg_local, str) and advantage_cfg_local.lower() == "baseline_rescaled":
            advantage_cfg_local = {"type": advantage_cfg_local, "params": {}}
        if isinstance(advantage_cfg_local, dict):
            adv_type = str(advantage_cfg_local.get("type", "")).lower()
            if adv_type == "baseline_rescaled":
                params = advantage_cfg_local.setdefault("params", {})
                for key in ("calibration_path", "problem", "dim", "type_instance", "top_k", "h_top_k"):
                    if key not in params and key in kernel_config_local:
                        params[key] = kernel_config_local.get(key)
                if "dim" not in params:
                    params["dim"] = self.N
                if "problem" not in params:
                    params["problem"] = getattr(self, "problem_type", None)
        self.advantage_strategy = AdvantageFactory.from_config(advantage_cfg_local)
        self.kernel_config = kernel_config_local
        self.kernel_name = str(self.kernel_config.get("name", "hk")).lower()
        self.kernel_params = {}
        self.prob_eps_clamp = float(self.kernel_config.get("prob_eps_clamp", 1e-3))
        self.debug_svgd = bool(self.kernel_config.get("debug_svgd", True))
        self.debug_every = int(self.kernel_config.get("debug_every", 10))
        self._debug_step = 0
        self._last_debug_stats = None
        self._last_phi_stats = None

        # expose agent-level info for monitoring code (hamming/KL, leaderboard)
        self.agent_lambdas = [self.lambda_per_agent for _ in range(self.M)]
        self.agents = []

        # interaction SVGD 
        kernel_impl = self._build_svgd_kernel(self.kernel_name, self.kernel_params)
        if self.mask is not None:
            setattr(kernel_impl, "mask", self.mask)
        self.svgd = SVGD(kernel_impl, gamma=self.svgd_gamma, no_repulsion=self.no_repulsion)
        self.theta_history = []
        self.kernel_metric_history = []

        # Paramètres appris : theta (nb_instances, M, N) initialisé dans reset
        self.theta = None
        self.nb_instances = 0
        self.latest_advantages = None
        self.probs = None

        # Buffers (B, M)
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
            logits = self.theta
            if self.mask is not None:
                logits = logits.masked_fill(self.mask == 0, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=1.0 / float(probs.size(-1)))
            if self.mask is not None:
                probs = probs * self.mask
                denom = probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                probs = probs / denom
            probs = torch.clamp(probs, self.prob_eps_clamp, 1.0 - self.prob_eps_clamp)
            if self.mask is not None:
                probs = probs * self.mask
                denom = probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                probs = probs / denom
            else:
                probs = probs / probs.sum(dim=-1, keepdim=True)
            self.probs = probs
            return probs  # (B, M, N, D)
        probs = torch.sigmoid(self.theta)
        self.probs = torch.clamp(torch.nan_to_num(probs, nan=0.5), self.prob_eps_clamp, 1 - self.prob_eps_clamp)
        return probs  # (B, M, N)

    def reset_learned_parameters(self, nb_instances):
        self.nb_instances = nb_instances

        # theta : (B, M, N) or (B, M, N, D)
        init_sigma = 0.1
        if self.use_categorical:
            max_dim = self.max_dim or 3
            init_theta = torch.randn(
                (nb_instances, self.M, self.N, max_dim), device=self.device
            ) * init_sigma
        else:
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
            probs = self.probs  # (B, M, N, D)
            D = probs.size(-1)
            flat = probs.reshape(-1, D)  # (B*M*N, D)
            samples_flat = torch.multinomial(flat, num_samples=λa, replacement=True)  # (B*M*N, λa)
            samples_agents = samples_flat.view(B, M, N, λa).permute(0, 1, 3, 2)  # (B, M, λa, N)
            samples = samples_agents.reshape(B, λ_total, N).unsqueeze(-1).float()
            return samples

        # u : (B, M, λa, N)
        u = torch.rand((B, M, λa, N), device=self.device)
        samples_agents = (u < self.probs.unsqueeze(2)).float()  # (B, M, λa, N)

        # On concatène tous les agents sur la dimension λ : (B, λ, N, 1)
        # on tire tout les lamdba en une fois pour eviter les boucles
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
            greedy = torch.argmax(probs, dim=-1)  # (B, M, N)
            return greedy.unsqueeze(-1).float()
        greedy = (probs >= 0.5).float()  # (B, M, N)
        return greedy.unsqueeze(-1)

    def updateDistribution(self, solutionList, scoreList):
        """
        Applique la mise à jour REINFORCE suivie de SVGD entre agents (si activé).
        """
        self._debug_step += 1
        # RL update (vectorisé sur (B, M))
        total_loss = self._updateDistribution_REINFORCE(solutionList, scoreList)
        # SVGD entre agents 
        self._apply_svgd()
        if self.enable_visualization:
            self._record_theta()
        if self._should_debug():
            self._print_debug()
        return total_loss

    # =======================
    #   REINFORCE vectorisé
    # =======================

    def _updateDistribution_REINFORCE(self, solutionList, scoreList):
        B, M, N = self.nb_instances, self.M, self.N
        λa = self.lambda_per_agent
        BM = B * M
        indivduals = solutionList.view(BM, λa, N)
        fitness = scoreList.view(BM, λa)
        baseline = self.baseline.view(BM)

        if self.baseline.numel() == 0:
            baseline = torch.zeros(BM, device=self.device)

        if self.use_categorical:
            if self.probs is None:
                self.forward()
            D = self.probs.size(-1)
            theta = self.theta.view(BM, N, D)
            all_Pi_Theta = self.probs.view(BM, N, D)  # (BM, N, D)
            all_Pi_Theta_expanded = all_Pi_Theta.unsqueeze(1).expand(-1, λa, -1, -1)  # (BM, λa, N, D)
            log_probs = torch.log(all_Pi_Theta_expanded + 1e-10)
            indices = indivduals.long().unsqueeze(-1)  # (BM, λa, N, 1)
            log_Pi = log_probs.gather(-1, indices).squeeze(-1).sum(dim=2)  # (BM, λa)
        else:
            theta = self.theta.view(BM, N)
            all_Pi_Theta = self.probs.view(BM, N)  # (BM, N)
            all_Pi_Theta_expanded = all_Pi_Theta.unsqueeze(1).expand(-1, λa, -1)  # (BM, λa, N)

            Pi_selected = torch.where(
                indivduals == 1.0,
                all_Pi_Theta_expanded,
                1.0 - all_Pi_Theta_expanded,
            )  # (BM, λa, N)
            log_Pi = torch.log(Pi_selected + 1e-10).sum(dim=2)  # (BM, λa)
        advantages = self.advantage_strategy.compute(
            fitness=fitness,
            baseline=baseline,
            theta=theta,
            indivduals=indivduals,
            probs=all_Pi_Theta_expanded,
            nb_instances=B,
            num_agents=M,
        )  # (BM, λa)
        loss_per_instance = torch.mean(advantages * log_Pi, dim=1)  # (BM,)
        loss = loss_per_instance.sum()
        if self.debug_svgd:
            with torch.no_grad():
                self._last_debug_stats = {
                    "loss_mean": float(loss_per_instance.mean().item()),
                    "adv_mean": float(advantages.mean().item()),
                    "adv_std": float(advantages.std().item()),
                    "adv_min": float(advantages.min().item()),
                    "adv_max": float(advantages.max().item()),
                    "fit_mean": float(fitness.mean().item()),
                    "fit_std": float(fitness.std().item()),
                    "fit_min": float(fitness.min().item()),
                    "fit_max": float(fitness.max().item()),
                    "baseline_mean": float(baseline.mean().item()) if baseline.numel() else float("nan"),
                    "baseline_std": float(baseline.std().item()) if baseline.numel() > 1 else 0.0,
                    "logpi_mean": float(log_Pi.mean().item()),
                    "logpi_min": float(log_Pi.min().item()),
                    "logpi_max": float(log_Pi.max().item()),
                    "prob_mean": float(all_Pi_Theta.mean().item()),
                    "prob_min": float(all_Pi_Theta.min().item()),
                    "prob_max": float(all_Pi_Theta.max().item()),
                    "adv_nan": bool(torch.isnan(advantages).any().item()),
                    "prob_nan": bool(torch.isnan(all_Pi_Theta).any().item()),
                }
        with torch.no_grad():
            reshaped_adv = advantages.detach().view(B, M, λa)
            per_instance = reshaped_adv.view(B, self.lambda_)
            self.latest_advantages = per_instance.cpu()

        grad_theta, = torch.autograd.grad(loss, self.theta, create_graph=False, retain_graph=True)
        self.last_theta_grad = grad_theta.detach().clone()
        if self.debug_svgd and self._last_debug_stats is not None:
            with torch.no_grad():
                self._last_debug_stats["theta_grad_norm"] = float(grad_theta.norm().item())

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
            phi = self.svgd.phi(theta, score, probs=self.probs)  # (B, M, N)
            kernel_stats = self.svgd.get_last_kernel_stats()
            if kernel_stats:
                self.kernel_metric_history.append(kernel_stats)
            if self.debug_svgd:
                with torch.no_grad():
                    self._last_phi_stats = {
                        "phi_mean": float(phi.mean().item()),
                        "phi_std": float(phi.std().item()),
                        "phi_norm": float(phi.norm().item()),
                        "phi_max_abs": float(phi.abs().max().item()),
                        "phi_nan": bool(torch.isnan(phi).any().item()),
                    }

        with torch.no_grad():
            self.theta += self.epsilon_svgd * phi
            self.probs = None

    def _should_debug(self) -> bool:
        if not self.debug_svgd:
            return False
        if self.debug_every < 1:
            return False
        if self._debug_step == 1:
            return True
        return (self._debug_step % self.debug_every) == 0

    @staticmethod
    def _fmt(val) -> str:
        if val is None:
            return "na"
        if isinstance(val, bool):
            return "1" if val else "0"
        if isinstance(val, (int, float)):
            if isinstance(val, float) and not math.isfinite(val):
                return str(val)
            return f"{float(val):.4g}"
        return str(val)

    def _print_debug(self) -> None:
        stats = self._last_debug_stats or {}
        phi_stats = self._last_phi_stats or {}
        msg = (
            f"[SVGD_EDA][step {self._debug_step}] "
            f"loss={self._fmt(stats.get('loss_mean'))} "
            f"adv(m/s/min/max)={self._fmt(stats.get('adv_mean'))}/"
            f"{self._fmt(stats.get('adv_std'))}/"
            f"{self._fmt(stats.get('adv_min'))}/"
            f"{self._fmt(stats.get('adv_max'))} "
            f"fit(m/s/min/max)={self._fmt(stats.get('fit_mean'))}/"
            f"{self._fmt(stats.get('fit_std'))}/"
            f"{self._fmt(stats.get('fit_min'))}/"
            f"{self._fmt(stats.get('fit_max'))} "
            f"prob(mn/mx)={self._fmt(stats.get('prob_min'))}/{self._fmt(stats.get('prob_max'))} "
            f"logpi(mn/mx)={self._fmt(stats.get('logpi_min'))}/{self._fmt(stats.get('logpi_max'))} "
            f"grad_norm={self._fmt(stats.get('theta_grad_norm'))} "
            f"phi_norm={self._fmt(phi_stats.get('phi_norm'))} "
            f"eps={self._fmt(self.epsilon_svgd)} gamma={self._fmt(self.svgd.gamma)} "
            f"kernel={self.kernel_name} no_interact={int(self.no_interact)} "
            f"no_repulsion={int(self.no_repulsion)} "
            f"nan_adv={self._fmt(stats.get('adv_nan'))} nan_prob={self._fmt(stats.get('prob_nan'))} "
            f"nan_phi={self._fmt(phi_stats.get('phi_nan'))}"
        )
        print(msg, flush=True)

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

    # =======================
    #   Visualisation
    # =======================

    def _record_theta(self):
        if self.theta is None or self.nb_instances <= 0:
            return []
        with torch.no_grad():
            probs = self.probs if self.probs is not None else self.forward()
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
        if self.no_interact or kernel in ("no_interact", "no-interact", "identity", "none"):
            return NoInteractKernel()
        if kernel in ("hk", "hamming", "hammingkernel"):
            return HammingKernel()
        if kernel == "ppk":
            return PPK()
        if kernel == "rbf":
            bandwith_kernel = self.kernel_config.get("bandwith_kernel")
            return RBF(bandwith_kernel=bandwith_kernel)
        if kernel == "pk":
            bandwith_kernel = self.kernel_config.get("bandwith_kernel")
            return ProbabilityKernel(bandwith_kernel=bandwith_kernel)
        if kernel == "jsd":
            bandwith_kernel = self.kernel_config.get("bandwith_kernel")
            return JSD(bandwith_kernel=bandwith_kernel)
        if kernel in ("fr", "fisherrao", "fisher_rao", "fisher-rao"):
            bandwith_kernel = self.kernel_config.get("bandwith_kernel")
            return FisherRaoKernel(bandwith_kernel=bandwith_kernel)
        raise ValueError(
            f"Unsupported kernel '{kernel_name}'. Available kernels: hk, ppk, rbf, pk, jsd, fr, no_interact."
        )

    def initialize_from_dataset(self, x_data, max_samples: int = 50000, noise_std: float = 0.01) -> bool:
        """
        Initialize categorical logits from empirical per-position distribution.
        Works with tokens (B, N) or onehot (B, N, D).
        """
        if not self.use_categorical or self.theta is None:
            return False
        try:
            x_np = np.asarray(x_data)
        except Exception:
            return False
        if x_np.ndim not in (2, 3):
            return False
        if x_np.shape[0] > max_samples:
            idx = np.random.choice(x_np.shape[0], size=max_samples, replace=False)
            x_np = x_np[idx]
        if x_np.ndim == 3:
            if self.max_dim is None or x_np.shape[-1] != self.max_dim:
                return False
            p = x_np.mean(axis=0).astype(np.float32)
        else:
            if self.max_dim is None:
                return False
            n_samples, n_dim = x_np.shape
            if n_dim != self.N:
                return False
            p = np.zeros((n_dim, self.max_dim), dtype=np.float32)
            for j in range(n_dim):
                counts = np.bincount(x_np[:, j].astype(np.int64), minlength=self.max_dim).astype(np.float32)
                total = float(counts.sum())
                if total > 0:
                    p[j] = counts / total
        p = np.clip(p, self.prob_eps_clamp, 1.0)
        p = p / np.sum(p, axis=-1, keepdims=True)
        logits = np.log(p)
        base = torch.as_tensor(logits, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            expanded = base.unsqueeze(0).unsqueeze(0).expand(self.nb_instances, self.M, -1, -1)
            if noise_std and noise_std > 0:
                expanded = expanded + (noise_std * torch.randn_like(expanded))
            self.theta.copy_(expanded)
            self.probs = None
        self._refresh_agent_views()
        return True
