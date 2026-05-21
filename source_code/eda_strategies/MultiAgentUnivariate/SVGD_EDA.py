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
        self.gradient_memory = None
        self.agent_fitness_memory = None
        self.visited_mask = None
        self.current_active_indices = None
        self.current_active_mask = None
        self.current_active_probs = None
        self.l_active = self.M
        self.partial_updates_enabled = False

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

        self.theta_history = []
        self.kernel_metric_history = []
        self.last_theta_grad = None
        self.gradient_memory = torch.zeros_like(self.theta.detach())
        self.agent_fitness_memory = torch.zeros((self.M,), dtype=torch.float32, device=self.device)
        self.visited_mask = torch.zeros((nb_instances, self.M), dtype=torch.bool, device=self.device)
        self.current_active_indices = None
        self.current_active_mask = None
        self.current_active_probs = None
        if self.enable_visualization:
            self._record_theta()
        self.latest_advantages = None
        self.probs = None

    def configure_partial_updates(self, l_active=None):
        if l_active is None:
            l_active = self.M
        l_active = int(l_active)
        if l_active < 1 or l_active > self.M:
            raise ValueError(f"l_active must be in [1, {self.M}], got {l_active}.")
        if l_active < self.M and self.kernel_name != "rbf":
            raise ValueError(
                f"Partial particle updates are only supported with the rbf kernel, got '{self.kernel_name}'."
            )
        self.l_active = l_active
        self.partial_updates_enabled = bool(self.l_active < self.M)
        self.lambda_ = self.lambda_per_agent * self.l_active
        if self.partial_updates_enabled:
            self.agent_lambdas = []
        else:
            self.agent_lambdas = [self.lambda_per_agent for _ in range(self.M)]

    def _compute_probs_from_theta(self, theta):
        if self.use_categorical:
            logits = theta
            if self.mask is not None:
                if logits.dim() == self.mask.dim():
                    mask = self.mask
                else:
                    expand_shape = [logits.size(0), logits.size(1), logits.size(2), logits.size(3)]
                    mask = self.mask.expand(expand_shape)
                logits = logits.masked_fill(mask == 0, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=1.0 / float(probs.size(-1)))
            if self.mask is not None:
                if probs.dim() == self.mask.dim():
                    mask = self.mask
                else:
                    expand_shape = [probs.size(0), probs.size(1), probs.size(2), probs.size(3)]
                    mask = self.mask.expand(expand_shape)
                probs = probs * mask
                denom = probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                probs = probs / denom
            probs = torch.clamp(probs, self.prob_eps_clamp, 1.0 - self.prob_eps_clamp)
            if self.mask is not None:
                probs = probs * mask
                denom = probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                probs = probs / denom
            else:
                probs = probs / probs.sum(dim=-1, keepdim=True)
            return probs
        probs = torch.sigmoid(theta)
        return torch.clamp(torch.nan_to_num(probs, nan=0.5), self.prob_eps_clamp, 1 - self.prob_eps_clamp)

    def sample_solutions(self):
        """
        Génère (nb_instances, λ, N, 1) en une seule fois.

        Chaque agent possède son propre budget λ_agent (= lambda_per_agent),
        on échantillonne (B, M, λ_agent, N), puis on aplati en (B, λ_total, N, 1)
        avec λ_total = M * λ_agent.
        """
        B, M, N = self.nb_instances, self.M, self.N
        λa = self.lambda_per_agent
        active_count = self.l_active if self.partial_updates_enabled else M
        λ_total = λa * active_count

        if self.partial_updates_enabled:
            active_indices = torch.stack(
                [torch.randperm(M, device=self.device)[:active_count] for _ in range(B)],
                dim=0,
            )
            active_mask = torch.zeros((B, M), dtype=torch.bool, device=self.device)
            active_mask.scatter_(1, active_indices, True)
            self.current_active_indices = active_indices
            self.current_active_mask = active_mask

            if self.use_categorical:
                D = self.theta.size(-1)
                gather_index = active_indices.unsqueeze(-1).unsqueeze(-1).expand(B, active_count, N, D)
                active_theta = self.theta.gather(1, gather_index)
                active_probs = self._compute_probs_from_theta(active_theta)
                self.current_active_probs = active_probs
                gather_index = active_indices.unsqueeze(-1).unsqueeze(-1).expand(B, active_count, N, D)
                flat = active_probs.reshape(-1, D)
                samples_flat = torch.multinomial(flat, num_samples=λa, replacement=True)
                samples_agents = samples_flat.view(B, active_count, N, λa).permute(0, 1, 3, 2)
                return samples_agents.reshape(B, λ_total, N).unsqueeze(-1).float()

            gather_index = active_indices.unsqueeze(-1).expand(B, active_count, N)
            active_theta = self.theta.gather(1, gather_index)
            active_probs = self._compute_probs_from_theta(active_theta)
            self.current_active_probs = active_probs
            u = torch.rand((B, active_count, λa, N), device=self.device)
            samples_agents = (u < active_probs.unsqueeze(2)).float()
            return samples_agents.view(B, λ_total, N).unsqueeze(-1)

        self.probs = self.forward()
        self.current_active_probs = None

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
        """
        Applique la mise à jour REINFORCE suivie de SVGD entre agents (si activé).
        """
        self._debug_step += 1
        if self.partial_updates_enabled:
            total_loss = self._updateDistribution_REINFORCE_partial(solutionList, scoreList)
        else:
            total_loss = self._updateDistribution_REINFORCE(solutionList, scoreList)
        self._apply_svgd()
        if self.enable_visualization:
            self._record_theta()
        if self._should_debug():
            self._print_debug()
        return total_loss


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
            reshaped_adv = advantages.detach().view(B, M, λa)
            per_instance = reshaped_adv.view(B, self.lambda_)
            self.latest_advantages = per_instance.cpu()

        grad_theta, = torch.autograd.grad(loss, self.theta, create_graph=False, retain_graph=True)
        self.last_theta_grad = grad_theta.detach().clone()
        if self.debug_svgd and self._last_debug_stats is not None:
            with torch.no_grad():
                self._last_debug_stats["theta_grad_norm"] = float(grad_theta.norm().item())

        with torch.no_grad():
            baseline_new = fitness.mean(dim=1)         
            self.baseline = baseline_new.view(B, M)

        return loss_per_instance.mean()

    def _updateDistribution_REINFORCE_partial(self, solutionList, scoreList):
        if self.current_active_indices is None:
            raise RuntimeError("sample_solutions must be called before partial updateDistribution.")

        B, N = self.nb_instances, self.N
        l_active = self.l_active
        λa = self.lambda_per_agent
        active_BM = B * l_active
        active_indices = self.current_active_indices
        indivduals = solutionList.view(B, l_active, λa, N).reshape(active_BM, λa, N)
        fitness = scoreList.view(B, l_active, λa).reshape(active_BM, λa)
        baseline = self.baseline.gather(1, active_indices).reshape(active_BM)

        if self.use_categorical:
            D = self.theta.size(-1)
            gather_index = active_indices.unsqueeze(-1).unsqueeze(-1).expand(B, l_active, N, D)
            theta = self.theta.gather(1, gather_index).reshape(active_BM, N, D)
            active_probs = self._compute_probs_from_theta(theta.view(B, l_active, N, D)).reshape(active_BM, N, D)
            all_Pi_Theta_expanded = active_probs.unsqueeze(1).expand(-1, λa, -1, -1)
            log_probs = torch.log(all_Pi_Theta_expanded + 1e-10)
            indices = indivduals.long().unsqueeze(-1)
            log_Pi = log_probs.gather(-1, indices).squeeze(-1).sum(dim=2)
        else:
            gather_index = active_indices.unsqueeze(-1).expand(B, l_active, N)
            theta = self.theta.gather(1, gather_index).reshape(active_BM, N)
            active_probs = self._compute_probs_from_theta(theta.view(B, l_active, N)).reshape(active_BM, N)
            all_Pi_Theta_expanded = active_probs.unsqueeze(1).expand(-1, λa, -1)
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
            num_agents=l_active,
        )
        loss_per_instance = torch.mean(advantages * log_Pi, dim=1)
        loss = loss_per_instance.sum()

        grad_active, = torch.autograd.grad(loss, theta, create_graph=False)
        grad_active = grad_active.detach().view(B, l_active, *theta.shape[1:])

        if self.gradient_memory is None:
            self.gradient_memory = torch.zeros_like(self.theta.detach())
        if self.last_theta_grad is None:
            self.last_theta_grad = torch.zeros_like(self.theta.detach())

        with torch.no_grad():
            self.gradient_memory.scatter_(1, gather_index, grad_active)
            self.last_theta_grad.scatter_(1, gather_index, grad_active)
            baseline_new = scoreList.view(B, l_active, λa).mean(dim=2)
            for batch_idx in range(B):
                self.baseline[batch_idx, active_indices[batch_idx]] = baseline_new[batch_idx]
            self.latest_advantages = advantages.detach().view(B, l_active * λa).cpu()

        return loss_per_instance.mean()

    def get_latest_advantages(self):
        if self.latest_advantages is None:
            return None
        return self.latest_advantages.detach().cpu()

    def get_agent_fitness_snapshot(self, tensor_score):
        if tensor_score is None:
            return []
        if self.agent_fitness_memory is None or self.agent_fitness_memory.numel() != self.M:
            self.agent_fitness_memory = torch.zeros((self.M,), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if self.partial_updates_enabled:
                if self.current_active_indices is None:
                    return self.agent_fitness_memory.detach().cpu().tolist()
                B = tensor_score.size(0)
                λa = self.lambda_per_agent
                active_scores = tensor_score.view(B, self.l_active, λa).max(dim=2).values
                sums = torch.zeros((self.M,), dtype=torch.float32, device=tensor_score.device)
                counts = torch.zeros((self.M,), dtype=torch.float32, device=tensor_score.device)
                ones = torch.ones((self.l_active,), dtype=torch.float32, device=tensor_score.device)
                for batch_idx in range(B):
                    active_idx = self.current_active_indices[batch_idx]
                    sums.index_add_(0, active_idx, active_scores[batch_idx])
                    counts.index_add_(0, active_idx, ones)
                mask = counts > 0
                if mask.any():
                    self.agent_fitness_memory[mask] = sums[mask] / counts[mask]
                return self.agent_fitness_memory.detach().cpu().tolist()

            B = tensor_score.size(0)
            λa = self.lambda_per_agent
            scores = tensor_score.view(B, self.M, λa).max(dim=2).values.mean(dim=0)
            self.agent_fitness_memory.copy_(scores)
            return self.agent_fitness_memory.detach().cpu().tolist()


    def toString(self):
        return f"MultiAgent_Collaborative_M{self.M}_lambdaPerAgent{self.lambda_per_agent}"

    def _apply_svgd(self):
        """
        Applique un pas SVGD instance par instance en se basant sur les directions RL observées.
        Utilise self.last_theta_grad comme direction RL : (B, M, N)
        """
        if self.last_theta_grad is None:
            return

        if self.partial_updates_enabled:
            self._apply_svgd_partial()
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

    def _apply_svgd_partial(self):
        if self.current_active_indices is None or self.current_active_mask is None:
            return
        if self.gradient_memory is None:
            return

        theta = self.theta
        B = self.nb_instances
        l_active = self.l_active
        N = self.N
        M = self.M
        active_indices = self.current_active_indices  # [B, l_active]

        # Support set: toutes les particules ayant eu de l'activité (visitées OU actives)
        eligible_mask = self.visited_mask | self.current_active_mask  # [B, M]

        if theta.dim() == 4:
            # Categorical (NK3): theta [B, M, N, D]
            D = theta.size(-1)

            query_index = active_indices.unsqueeze(-1).unsqueeze(-1).expand(B, l_active, N, D)
            query_theta = theta.gather(1, query_index)  # [B, l_active, N, D]

            # Support = toutes les M particules ; eligible_mask filtre les inéligibles dans phi
            support_theta = theta.detach().unsqueeze(1).expand(B, l_active, M, N, D).contiguous()
            support_score = self.gradient_memory.unsqueeze(1).expand(B, l_active, M, N, D).contiguous()

            scatter_index = active_indices.unsqueeze(-1).unsqueeze(-1).expand(B, l_active, N, D)
        else:
            # Binary: theta [B, M, N]
            query_index = active_indices.unsqueeze(-1).expand(B, l_active, N)
            query_theta = theta.gather(1, query_index)  # [B, l_active, N]

            # Support = toutes les M particules ; eligible_mask filtre les inéligibles dans phi
            support_theta = theta.detach().unsqueeze(1).expand(B, l_active, M, N).contiguous()
            support_score = self.gradient_memory.unsqueeze(1).expand(B, l_active, M, N).contiguous()

            scatter_index = active_indices.unsqueeze(-1).expand(B, l_active, N)

        full_phi = torch.zeros_like(theta)
        with torch.enable_grad():
            phi = self.svgd.phi(
                query_theta,
                support_score,
                probs=None,
                support_thetas=support_theta,
                support_probs=None,
                support_mask=eligible_mask,
            )  # [B, l_active, N] or [B, l_active, N, D]

        kernel_stats = self.svgd.get_last_kernel_stats()
        if kernel_stats:
            self.kernel_metric_history.append(kernel_stats)

        with torch.no_grad():
            full_phi.scatter_(1, scatter_index, phi.detach())
            self.theta += self.epsilon_svgd * full_phi
            self.visited_mask |= self.current_active_mask
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
