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
        adaptive_lambda=False,
        lr_lambda=0.1,
        lambda_range=0.6,
    ):
        self.M = M
        self.N = N
        self.lambda_per_agent = int(lambda_)
        self.total_lambda = self.lambda_per_agent * self.M
        Abstract_EDA.__init__(self, N, self.total_lambda, device)
        nn.Module.__init__(self)

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

        self.agent_lambdas = [self.lambda_per_agent for _ in range(self.M)]
        self.lambda_per_agent_init = self.lambda_per_agent
        self.adaptive_lambda = bool(adaptive_lambda)
        if self.adaptive_lambda:
            self.lr_lambda    = float(lr_lambda)
            self.lambda_range = float(lambda_range)
            self.delta_base   = round(self.lr_lambda * self.lambda_per_agent_init)
            self.lambda_min_per_agent = max(1, round((1 - self.lambda_range) * self.lambda_per_agent_init))
            self.lambda_max_per_agent = round((1 + self.lambda_range) * self.lambda_per_agent_init)
        self.agents = []

        kernel_impl = self._build_svgd_kernel(self.kernel_name, self.kernel_params)
        if self.mask is not None:
            setattr(kernel_impl, "mask", self.mask)
        self.svgd = SVGD(kernel_impl, gamma=self.svgd_gamma, no_repulsion=self.no_repulsion)
        self.theta_history = []
        self.kernel_metric_history = []

        self.theta = None
        self.nb_instances = 0
        self.latest_advantages = None
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

        if self.adaptive_lambda:
            self.agent_lambdas = [self.lambda_per_agent_init] * self.M
            self.agent_lambdas_bi = torch.full(
                (nb_instances, self.M), self.lambda_per_agent_init, dtype=torch.int32, device=self.device
            )

        self.theta_history = []
        self.kernel_metric_history = []
        self.last_theta_grad = None
        self.latest_advantages = None
        if self.enable_visualization:
            self._record_theta()
        self.latest_advantages = None
        self.probs = None

    def sample_solutions(self):
        """
        Génère (nb_instances, λ_total, N, 1).

        Adaptive : toujours lambda_max_per_agent solutions par agent → les gagnants
                   utilisent toutes leurs solutions pour le gradient.
        Non-adaptive fast path : opération matricielle unique (B, M, λa, N).
        Non-adaptive slow path : boucle par agent (lambdas hétérogènes).
        """
        B, M, N = self.nb_instances, self.M, self.N

        self.probs = self.forward()

        if self.adaptive_lambda and hasattr(self, "agent_lambdas_bi"):
            # Génère exactement agent_lambdas_bi[b,m] solutions pour chaque (instance, agent).
            # Total par instance = M × lambda_per_agent_init (conservation de la somme).
            # Boucle sur (b, m) : overhead Python acceptable, pas d'éval gaspillée.
            output = torch.zeros(B, self.total_lambda, N, device=self.device)
            cum = torch.zeros(B, dtype=torch.int32)
            for m in range(M):
                for b in range(B):
                    lam_bm  = int(self.agent_lambdas_bi[b, m].item())
                    off_bm  = int(cum[b].item())
                    probs_bm = self.probs[b, m]                                # (N,)
                    if self.use_categorical:
                        D = probs_bm.size(-1)
                        idx = torch.multinomial(probs_bm.unsqueeze(0).expand(N, -1)
                                                if probs_bm.dim() == 1 else probs_bm,
                                                num_samples=lam_bm, replacement=True)
                        # Simplifié : arrondi depuis probs pour catégoriel
                        flat = probs_bm.reshape(N, -1)
                        idx  = torch.multinomial(flat, num_samples=lam_bm, replacement=True)
                        output[b, off_bm:off_bm + lam_bm] = idx.t().float()
                    else:
                        u = torch.rand(lam_bm, N, device=self.device)
                        output[b, off_bm:off_bm + lam_bm] = (u < probs_bm.unsqueeze(0)).float()
                    cum[b] += lam_bm
            return output.unsqueeze(-1)

        # Non-adaptive : fast path si lambdas uniformes
        if len(set(self.agent_lambdas)) == 1:
            λa = self.agent_lambdas[0]
            if self.use_categorical:
                D = self.probs.size(-1)
                flat = self.probs.reshape(-1, D)
                samples_flat = torch.multinomial(flat, num_samples=λa, replacement=True)
                samples_agents = samples_flat.view(B, M, N, λa).permute(0, 1, 3, 2)
                return samples_agents.reshape(B, self.total_lambda, N).unsqueeze(-1).float()
            u = torch.rand((B, M, λa, N), device=self.device)
            samples_agents = (u < self.probs.unsqueeze(2)).float()
            return samples_agents.view(B, self.total_lambda, N).unsqueeze(-1)

        # Non-adaptive slow path : lambdas hétérogènes
        samples_list = []
        for m, lam_m in enumerate(self.agent_lambdas):
            probs_m = self.probs[:, m]
            if self.use_categorical:
                D = probs_m.size(-1)
                flat = probs_m.reshape(B * N, D)
                idx = torch.multinomial(flat, num_samples=lam_m, replacement=True)
                s_m = idx.view(B, N, lam_m).permute(0, 2, 1).float()
            else:
                u = torch.rand((B, lam_m, N), device=self.device)
                s_m = (u < probs_m.unsqueeze(1)).float()
            samples_list.append(s_m)
        return torch.cat(samples_list, dim=1).unsqueeze(-1)

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
        """
        Applique la mise à jour REINFORCE suivie de SVGD entre agents (si activé).
        """
        self._debug_step += 1
        total_loss = self._updateDistribution_REINFORCE(solutionList, scoreList)
        self._apply_svgd()
        self.adapt_lambdas()
        if self.enable_visualization:
            self._record_theta()
        if self._should_debug():
            self._print_debug()
        return total_loss


    def _updateDistribution_REINFORCE(self, solutionList, scoreList):
        B, M, N = self.nb_instances, self.M, self.N
        BM = B * M
        mask = None

        if self.adaptive_lambda and hasattr(self, "agent_lambdas_bi"):
            # Chemin adaptatif : chaque (b, m) a exactement agent_lambdas_bi[b,m] solutions
            # placées à une position variable dans solutionList.
            # On reconstruit un tenseur paddé (BM, λa_max, N) + masque pour REINFORCE.
            λa = self.lambda_max_per_agent
            indivduals = solutionList.new_zeros(BM, λa, N)
            fitness    = torch.zeros(BM, λa, device=self.device)
            mask       = torch.zeros(BM, λa, device=self.device)
            cum = torch.zeros(B, dtype=torch.int32)
            for m in range(M):
                for b in range(B):
                    lam_bm  = int(self.agent_lambdas_bi[b, m].item())
                    off_bm  = int(cum[b].item())
                    row     = b * M + m
                    indivduals[row, :lam_bm] = solutionList[b, off_bm:off_bm + lam_bm, :, 0]
                    scores_bm = scoreList[b, off_bm:off_bm + lam_bm]
                    fitness[row, :lam_bm]    = scores_bm
                    mask[row, :lam_bm]       = 1.0
                    if lam_bm < λa:
                        fitness[row, lam_bm:] = scores_bm.mean()
                    cum[b] += lam_bm
        elif len(set(self.agent_lambdas)) == 1:
            # Fast path non-adaptatif : lambdas uniformes — reshape direct
            λa = self.agent_lambdas[0]
            indivduals = solutionList.view(BM, λa, N)
            fitness = scoreList.view(BM, λa)
        else:
            # Slow path non-adaptatif : lambdas hétérogènes entre agents
            λa = max(self.agent_lambdas)
            indivduals = solutionList.new_zeros(BM, λa, N)
            fitness = torch.zeros(BM, λa, device=self.device)
            mask = torch.zeros(BM, λa, device=self.device)
            offset = 0
            for m, lam_m in enumerate(self.agent_lambdas):
                sols_m   = solutionList[:, offset:offset + lam_m, :, 0]
                scores_m = scoreList[:, offset:offset + lam_m]
                indivduals[m::M, :lam_m, :] = sols_m
                fitness[m::M, :lam_m]       = scores_m
                mask[m::M, :lam_m]          = 1.0
                if lam_m < λa:
                    mean_m = scores_m.mean(dim=1)
                    fitness[m::M, lam_m:] = mean_m.unsqueeze(1).expand(-1, λa - lam_m)
                offset += lam_m

        baseline = self.baseline.view(BM)
        if self.baseline.numel() == 0:
            baseline = torch.zeros(BM, device=self.device)

        if self.use_categorical:
            if self.probs is None:
                self.forward()
            D = self.probs.size(-1)
            theta = self.theta.view(BM, N, D)
            all_Pi_Theta = self.probs.view(BM, N, D)
            all_Pi_Theta_expanded = all_Pi_Theta.unsqueeze(1).expand(-1, λa, -1, -1)
            log_probs = torch.log(all_Pi_Theta_expanded + 1e-10)
            indices = indivduals.long().unsqueeze(-1)
            log_Pi = log_probs.gather(-1, indices).squeeze(-1).sum(dim=2)
        else:
            theta = self.theta.view(BM, N)
            all_Pi_Theta = self.probs.view(BM, N)
            all_Pi_Theta_expanded = all_Pi_Theta.unsqueeze(1).expand(-1, λa, -1)
            Pi_selected = torch.where(
                indivduals == 1.0,
                all_Pi_Theta_expanded,
                1.0 - all_Pi_Theta_expanded,
            )
            log_Pi = torch.log(Pi_selected + 1e-10).sum(dim=2)

        advantages = self.advantage_strategy.compute(
            fitness=fitness,
            baseline=baseline,
            theta=theta,
            indivduals=indivduals,
            probs=all_Pi_Theta_expanded,
            nb_instances=B,
            num_agents=M,
        )

        if mask is not None:
            valid_counts = mask.sum(dim=1).clamp(min=1)
            loss_per_instance = (advantages * log_Pi * mask).sum(dim=1) / valid_counts
        else:
            loss_per_instance = torch.mean(advantages * log_Pi, dim=1)

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
            if mask is not None:
                # Per-instance : garder le tenseur complet (B*M, λa), masque inclus
                self.latest_advantages = advantages.detach().cpu()
            else:
                self.latest_advantages = advantages.detach().view(B, M, λa).view(B, self.lambda_).cpu()

        grad_theta, = torch.autograd.grad(loss, self.theta, create_graph=False, retain_graph=True)
        self.last_theta_grad = grad_theta.detach().clone()
        if self.debug_svgd and self._last_debug_stats is not None:
            with torch.no_grad():
                self._last_debug_stats["theta_grad_norm"] = float(grad_theta.norm().item())

        with torch.no_grad():
            if mask is not None:
                valid_counts_bm = mask.sum(dim=1).clamp(min=1)
                baseline_new = (fitness * mask).sum(dim=1) / valid_counts_bm
            else:
                baseline_new = fitness.mean(dim=1)
            self.baseline = baseline_new.view(B, M)

        return loss_per_instance.mean()

    def get_latest_advantages(self):
        if self.latest_advantages is None:
            return None
        return self.latest_advantages.detach().cpu()

    def adapt_lambdas(self):
        """
        Redistribution du budget λ par instance, convergence vers une cible linéaire.

        Les cibles sont linéairement espacées de lambda_max (rang 1) à lambda_min (rang M).
        Pour chaque paire (rang k ↔ rang M-k), on transfère au plus delta_base unités,
        limité par la distance restante à la cible de chaque agent.
        Le total Σ_m agent_lambdas_bi[b,m] reste constant par instance.
        """
        if not self.adaptive_lambda or self.baseline.numel() == 0:
            return
        if not hasattr(self, "agent_lambdas_bi"):
            return

        B, M = self.baseline.shape
        lam_max = self.lambda_max_per_agent
        lam_min = self.lambda_min_per_agent

        # Cibles linéaires : rang 0 (meilleur) → lam_max, rang M-1 (pire) → lam_min
        targets = [
            round(lam_min + (lam_max - lam_min) * (M - 1 - k) / max(M - 1, 1))
            for k in range(M)
        ]

        for b in range(B):
            scores = self.baseline[b].detach()
            sorted_indices = torch.argsort(scores, descending=True).tolist()

            for k in range(M // 2):
                winner = sorted_indices[k]
                loser  = sorted_indices[M - 1 - k]

                if scores[winner].item() <= scores[loser].item():
                    continue

                needed_gain = max(0, targets[k]       - int(self.agent_lambdas_bi[b, winner].item()))
                needed_give = max(0, int(self.agent_lambdas_bi[b, loser].item()) - targets[M - 1 - k])

                transfer = min(self.delta_base, needed_gain, needed_give)
                if transfer > 0:
                    self.agent_lambdas_bi[b, winner] += transfer
                    self.agent_lambdas_bi[b, loser]  -= transfer

    def toString(self):
        base = f"MultiAgent_Collaborative_M{self.M}_lambdaPerAgent{self.lambda_per_agent}"
        if self.adaptive_lambda:
            return base + f"_adaptiveLambda_lr{self.lr_lambda}"
        return base

    def _apply_svgd(self):
        """
        Applique un pas SVGD instance par instance en se basant sur les directions RL observées.
        Utilise self.last_theta_grad comme direction RL : (B, M, N)
        """
        if self.last_theta_grad is None:
            return

        theta = self.theta             
        score = self.last_theta_grad.detach()                                    

        with torch.enable_grad():
            phi = self.svgd.phi(theta, score, probs=self.probs)             
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
        return

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
            return []
        with torch.no_grad():
            probs = self.probs if self.probs is not None else self.forward()
        probs_final = [probs[:, m, :] for m in range(self.M)]                            
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
