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
from eda_strategies.MultiAgentUnivariate.advantage.rank_weighted import PerAgentRankWeightedAdvantage


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
        # PPO hyperparameters
        ppo_active=False,      # True => PPO (K époques), False => REINFORCE pur
        ppo_epochs=1,          # K : nombre d'époques internes (ignoré si ppo_active=False)
        ppo_mode='clip',       # 'clip' | 'kl' | 'trpo'
        clip_eps=0.2,          # ε_clip  (mode clip uniquement)
        kl_beta=1.0,           # β : pénalité KL  (mode kl uniquement)
        kl_target_kl=None,     # KL cible (float) => β adaptatif ; None => β fixe
        kl_beta_max=100.0,     # plafond de β (mode kl adaptatif uniquement)
        kl_beta_min=1e-4,      # plancher de β (mode kl adaptatif uniquement)
        trpo_kl_threshold=0.01,      # seuil max de KL_mean toléré par époque (mode trpo uniquement)
        trpo_backoff_max_tries=4,    # nb de tentatives de réduction du pas avant d'accepter tel quel (mode trpo)
        # Batch adaptatif par agent (inner-product test, Bollapragada/Byrd/Nocedal 2018)
        adaptive_batch=False,
        lambda_init=3,
        lambda_max=None,
        ip_tol=0.4,
    ):
        self.M = M
        self.N = N
        self.lambda_per_agent = int(lambda_)
        self.adaptive_batch_enabled = bool(adaptive_batch)
        self.lambda_init = int(lambda_init)
        self.ip_tol = float(ip_tol)
        if self.adaptive_batch_enabled:
            # lambda_per_agent devient le pire cas (lambda_max), utilisé par les
            # environnements appelants pour pré-allouer leurs tenseurs (taille de
            # population). Le vrai batch par agent est décidé dynamiquement par
            # adaptive_iteration() à chaque itération, entre lambda_init et lambda_max.
            self.lambda_max = int(lambda_max) if lambda_max is not None else self.lambda_per_agent
            self.lambda_per_agent = self.lambda_max
        else:
            self.lambda_max = int(lambda_max) if lambda_max is not None else self.lambda_per_agent
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

        # PPO
        self.ppo_active = bool(ppo_active)
        self.ppo_epochs = int(ppo_epochs)
        self.ppo_mode = str(ppo_mode)
        self.clip_eps = float(clip_eps)
        self.kl_beta_init = float(kl_beta)
        self.kl_target_kl = float(kl_target_kl) if kl_target_kl is not None else None
        self.kl_beta_max = float(kl_beta_max)
        self.kl_beta_min = float(kl_beta_min)
        self.kl_beta = self.kl_beta_init  # mutable, adapté si kl_target_kl est défini
        self.trpo_kl_threshold = float(trpo_kl_threshold)
        self.trpo_backoff_max_tries = int(trpo_backoff_max_tries)

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
        if self.adaptive_batch_enabled and not isinstance(self.advantage_strategy, PerAgentRankWeightedAdvantage):
            raise ValueError(
                "adaptive_batch=True n'est supporté que pour l'avantage "
                "'peragentrankweighted' pour l'instant."
            )
        if self.adaptive_batch_enabled and self.ppo_active:
            raise ValueError("adaptive_batch=True n'est supporté que pour REINFORCE (ppo_active=False) pour l'instant.")
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

        self.kl_beta = self.kl_beta_init  # reset β adaptatif à chaque nouvelle instance
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

    def _sample_from_probs(self, probs_m, count):
        """
        Tire `count` échantillons pour un agent, à partir de ses probs figées.
        probs_m : (B, N) binaire ou (B, N, D) catégoriel. Retourne (B, count, N, 1).
        """
        B = probs_m.size(0)
        if self.use_categorical:
            D = probs_m.size(-1)
            flat = probs_m.reshape(-1, D)
            samples_flat = torch.multinomial(flat, num_samples=count, replacement=True)
            samples = samples_flat.view(B, self.N, count).permute(0, 2, 1).unsqueeze(-1).float()
            return samples
        u = torch.rand((B, count, self.N), device=self.device)
        return (u < probs_m.unsqueeze(1)).float().unsqueeze(-1)

    def _score_function(self, indivduals, probs_m):
        """
        Forme close de ∇_theta log pi_theta(x) (Bernoulli/Categorical factorisés) :
        x - p (binaire) ou onehot(x) - p (catégoriel).
        indivduals : (B, lam, N). probs_m : (B, N) ou (B, N, D).
        """
        if self.use_categorical:
            D = probs_m.size(-1)
            onehot = torch.zeros(indivduals.shape + (D,), device=probs_m.device, dtype=probs_m.dtype)
            onehot.scatter_(-1, indivduals.long().unsqueeze(-1), 1.0)
            return onehot - probs_m.unsqueeze(1)
        return indivduals - probs_m.unsqueeze(1)

    def adaptive_iteration(self, evaluate_fn):
        """
        Une itération complète avec batch adaptatif par (instance, agent) — inner-product
        test, cf. Bollapragada/Byrd/Nocedal 2018. Chaque (instance b, agent m) décide sa
        propre taille de batch INDÉPENDAMMENT des autres instances : une instance au
        signal fort s'arrête tôt, une instance difficile continue seule, sans jamais
        imposer son rythme aux autres (cf. discussion : agréger sur B via un max faisait
        que toute instance difficile plafonnait tout le groupe au lambda_max).

        Pour garder un contrat externe inchangé (chaque agent occupe exactement
        lambda_max colonnes dans le tenseur retourné, comme avant — donc zéro
        changement requis dans les boucles appelantes), les instances arrêtées tôt
        sont complétées jusqu'à lambda_max par duplication de LEURS PROPRES échantillons
        déjà tirés et évalués (aucun appel oracle supplémentaire, aucune donnée
        fabriquée) ; le gradient et le critère, eux, n'utilisent jamais ces doublons —
        uniquement les échantillons réellement tirés pour cette instance.

        `evaluate_fn` : callable (tensor_solution (n, pop, N, 1)) -> tensor_score (n, pop),
        typiquement la même closure `_evaluate_population` que la boucle appelante
        utilise pour le chemin non-adaptatif (n = nombre d'instances encore actives
        ce tour-ci, pour ne jamais gaspiller d'évaluation sur une instance déjà satisfaite).

        Retourne (tensor_solution, tensor_score) au même format que
        sample_solutions() + evaluate_fn. updateDistribution est déjà appliqué en
        interne (theta est mis à jour) : ne pas le rappeler.
        """
        B, M, N = self.nb_instances, self.M, self.N
        self.probs = self.forward()

        per_agent_samples = []
        per_agent_fitness = []
        per_agent_grad = []
        per_agent_step_scale = []
        effective_lambda_per_agent = []
        per_agent_lambda_real = []

        for m in range(M):
            probs_m = self.probs[:, m, ...].detach()

            X = torch.zeros(B, self.lambda_max, N, 1, device=self.device)
            F = torch.zeros(B, self.lambda_max, device=self.device)
            lam_real = torch.zeros(B, dtype=torch.long, device=self.device)
            active = torch.ones(B, dtype=torch.bool, device=self.device)

            grad_shape = (B, N, self.max_dim) if self.use_categorical else (B, N)
            g_hat_final = torch.zeros(grad_shape, device=self.device)
            tr_sigma_final = torch.zeros(B, device=self.device)
            g_norm2_corr_final = torch.zeros(B, device=self.device)
            lam_req_final = torch.zeros(B, device=self.device)

            lam = 0
            while active.any():
                # increment <= 0 dès que lam atteint lambda_max : c'est le seul garde-fou nécessaire.
                increment = min(max(self.lambda_init, lam), self.lambda_max - lam)
                if increment <= 0:
                    break
                idx_active = active.nonzero(as_tuple=True)[0]
                new_indiv = self._sample_from_probs(probs_m[idx_active], increment)
                new_fitness = evaluate_fn(new_indiv, idx_active)  # évalué UNIQUEMENT pour les instances encore actives
                X[idx_active, lam:lam + increment] = new_indiv
                F[idx_active, lam:lam + increment] = new_fitness
                lam += increment

                # critère calculé UNIQUEMENT à partir des données propres à chaque instance active
                n_active = idx_active.numel()
                sub_fitness = F[idx_active, :lam]
                sub_indiv = X[idx_active, :lam].squeeze(-1)
                sub_probs = probs_m[idx_active]

                # a^(i) : poids PerAgentRankWeighted, reclassé sur l'ensemble accumulé de CETTE instance
                advantage = self.advantage_strategy.compute(
                    fitness=sub_fitness, nb_instances=n_active, num_agents=1,
                ).detach()
                score = self._score_function(sub_indiv, sub_probs)  # ∇_theta log pi_theta(x^(i)), forme close
                z = advantage.reshape(n_active, lam, *([1] * (score.dim() - 2))) * score  # z^(i) = a^(i) * score^(i)

                g_hat = z.mean(dim=1)  # ĝ = (1/λa) Σ z^(i) : estimateur du gradient
                diff = z - g_hat.unsqueeze(1)
                sq_norm_i = diff.flatten(start_dim=2).pow(2).sum(dim=2)
                # tr̂Σ = (1/(λa-1)) Σ ‖z^(i) - ĝ‖² : dispersion totale (variance MC), toutes directions confondues
                tr_sigma = sq_norm_i.sum(dim=1) / (lam - 1)

                g_norm2 = g_hat.flatten(start_dim=1).pow(2).sum(dim=1)  # ‖ĝ‖² brut (gonflé par le bruit)
                # ‖ĝ‖²_corr = max(‖ĝ‖² - tr̂Σ/λa, 0) : débiaisage de la norme (papier §3, "correction essentielle")
                g_norm2_corr = torch.clamp(g_norm2 - tr_sigma / lam, min=0.0)

                # ŝ^(i) = ⟨z^(i), ĝ⟩ / ‖ĝ‖² (norme BRUTE ici, cf. papier §3) : projection normalisée de
                # chaque échantillon sur la direction ĝ (vaut 1 en moyenne si ĝ est fiable).
                s_hat = (z * g_hat.unsqueeze(1)).flatten(start_dim=2).sum(dim=2) / g_norm2.clamp(min=1e-12).unsqueeze(1)
                var_s = s_hat.var(dim=1, unbiased=True)  # Var(ŝ^(i)) : variance de cette projection
                g_sigma_g = g_norm2.pow(2) * var_s  # ĝ^⊤Σĝ = ‖ĝ‖⁴ · Var(ŝ^(i)) : variance du gradient projetée sur ĝ

                # λa,req = ĝ^⊤Σĝ / (ip_tol² · ‖ĝ‖²_corr²) : inner-product test — taille de batch requise pour
                # que le signal (‖ĝ‖²_corr, débiaisé) domine le bruit projeté (ĝ^⊤Σĝ) avec la tolérance voulue.
                # +inf si aucun signal détectable (plateau, ‖ĝ‖²_corr=0).
                lam_req = torch.where(
                    g_norm2_corr > 0,
                    g_sigma_g / (self.ip_tol ** 2 * g_norm2_corr.clamp(min=1e-12).pow(2)),
                    torch.full_like(g_norm2_corr, float("inf")),
                )

                # arrêt indépendant par instance (repeat-until) : chaque instance active qui satisfait
                # SON PROPRE critère (ou atteint lambda_max) se fige ici, sans attendre les autres.
                newly_done = (lam_req <= lam) | (lam >= self.lambda_max)
                freeze_idx = idx_active[newly_done]
                if freeze_idx.numel() > 0:
                    g_hat_final[freeze_idx] = g_hat[newly_done]
                    tr_sigma_final[freeze_idx] = tr_sigma[newly_done]
                    g_norm2_corr_final[freeze_idx] = g_norm2_corr[newly_done]
                    lam_req_final[freeze_idx] = lam_req[newly_done]
                    lam_real[freeze_idx] = lam
                    active[freeze_idx] = False

            # Complétion à lambda_max par duplication (cyclique) des échantillons réels de
            # chaque instance — aucun appel oracle, aucune donnée inventée, juste pour garder
            # un bloc de taille fixe par agent (contrat externe inchangé pour les appelants).
            pad_idx = torch.arange(self.lambda_max, device=self.device).unsqueeze(0) % lam_real.clamp(min=1).unsqueeze(1)
            X_padded = torch.gather(X, 1, pad_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, 1))
            F_padded = torch.gather(F, 1, pad_idx)

            # capped/shrink : par instance, indépendamment — cf. discussion sur le besoin
            # d'indépendance entre instances (plus d'agrégation par max sur B).
            capped = (lam_real >= self.lambda_max) & (lam_req_final > lam_real)
            shrink = g_norm2_corr_final / (g_norm2_corr_final + tr_sigma_final / lam_real.clamp(min=1) + 1e-12)
            step_scale = torch.where(capped, shrink, torch.ones_like(shrink))

            per_agent_samples.append(X_padded)
            per_agent_fitness.append(F_padded)
            per_agent_grad.append(g_hat_final)
            per_agent_step_scale.append(step_scale)
            effective_lambda_per_agent.append(float(lam_real.float().mean().item()))
            per_agent_lambda_real.append(lam_real.detach().clone())

        tensor_solution = torch.cat(per_agent_samples, dim=1)
        tensor_score = torch.cat(per_agent_fitness, dim=1)
        # Contrat externe (slicing par bloc contigu dans les boucles appelantes) : chaque
        # agent occupe toujours exactement lambda_max colonnes, donc agent_lambdas reste
        # constant. La vraie taille par (instance, agent) est exposée séparément pour le
        # dashboard, où elle est bien plus informative qu'une constante.
        self.agent_lambdas = [self.lambda_max for _ in range(M)]
        self.last_effective_lambda_per_agent = effective_lambda_per_agent
        # (B, M) : vraie taille de batch par (instance, agent), pour le suivi dashboard
        # par instance (last_effective_lambda_per_agent n'expose que la moyenne sur B).
        self.last_lambda_per_instance = torch.stack(per_agent_lambda_real, dim=1)

        grad_theta = torch.stack(per_agent_grad, dim=1)  # (B, M, N[, D])
        self.last_theta_grad = grad_theta.detach()
        step_scale_full = torch.stack(per_agent_step_scale, dim=1)  # (B, M)
        step_scale_full = step_scale_full.view(B, M, *([1] * (grad_theta.dim() - 2)))

        with torch.no_grad():
            self.baseline = torch.stack([f.mean(dim=1) for f in per_agent_fitness], dim=1)
            self.latest_advantages = None  # ragged par instance : pas de vue plate simple ici

        self._apply_svgd(step_scale=step_scale_full)
        if self.enable_visualization:
            self._record_theta()

        return tensor_solution, tensor_score

    def updateDistribution(self, solutionList, scoreList):
        """Applique la mise à jour (REINFORCE ou PPO) suivie de SVGD entre agents."""
        self._debug_step += 1
        if self.ppo_active:
            total_loss = self._updateDistribution_PPO(solutionList, scoreList)
        else:
            total_loss = self._updateDistribution_REINFORCE(solutionList, scoreList)
            self._apply_svgd()
        if self.enable_visualization:
            self._record_theta()
        if self._should_debug():
            self._print_debug()
        return total_loss

    def _compute_kl(self, pi_old_full, pi_new_full):
        """
        KL(π_old || π_new) par (instance×agent, variable).
        binary:       pi (BM, N)      — probabilité de x=1
        categorical:  pi (BM, N, D)   — distribution sur les D catégories

        Retourne (kl_sum, kl_mean) :
        - kl_sum  : KL jointe (somme sur N, comme `surrogate`) — utilisée dans l'objectif,
          pour rester à la même échelle que le terme de récompense.
        - kl_mean : KL moyenne par position (BM×N) — utilisée uniquement pour comparer à
          `kl_target_kl`, qui garde une sémantique "KL moyenne par variable".
        """
        if self.use_categorical:
            from torch.distributions import Categorical, kl_divergence
            kl = kl_divergence(
                Categorical(probs=pi_old_full),
                Categorical(probs=pi_new_full),
            )  # (BM, N)
        else:
            from torch.distributions import Bernoulli, kl_divergence
            kl = kl_divergence(
                Bernoulli(probs=pi_old_full),
                Bernoulli(probs=pi_new_full),
            )  # (BM, N)
        kl_sum = kl.sum(dim=-1).sum()
        kl_mean = kl.mean()
        return kl_sum, kl_mean

    def _updateDistribution_REINFORCE(self, solutionList, scoreList):
        """
        Mise à jour REINFORCE pure : ∇_θ E[A · log π_θ(x)].
        Identique au comportement de la branche main (ppo_active=False).
        _apply_svgd() est appelé par updateDistribution après ce retour.
        """
        B, M, N = self.nb_instances, self.M, self.N
        λa = self.lambda_per_agent
        BM = B * M

        indivduals = solutionList.view(BM, λa, N)
        fitness = scoreList.view(BM, λa)
        baseline = self.baseline.view(BM) if self.baseline.numel() > 0 else torch.zeros(BM, device=self.device)

        if self.use_categorical:
            D = self.probs.size(-1)
            theta_flat = self.theta.view(BM, N, D)
            all_Pi_Theta = self.probs.view(BM, N, D)
            all_Pi_Theta_exp = all_Pi_Theta.unsqueeze(1).expand(-1, λa, -1, -1)
            log_probs = torch.log(all_Pi_Theta_exp + 1e-10)
            indices = indivduals.long().unsqueeze(-1)
            log_Pi = log_probs.gather(-1, indices).squeeze(-1).sum(dim=2)  # (BM, λa)
        else:
            theta_flat = self.theta.view(BM, N)
            all_Pi_Theta = self.probs.view(BM, N)
            all_Pi_Theta_exp = all_Pi_Theta.unsqueeze(1).expand(-1, λa, -1)
            Pi_selected = torch.where(indivduals == 1.0, all_Pi_Theta_exp, 1.0 - all_Pi_Theta_exp)
            log_Pi = torch.log(Pi_selected + 1e-10).sum(dim=2)  # (BM, λa)

        advantages = self.advantage_strategy.compute(
            fitness=fitness,
            baseline=baseline,
            theta=theta_flat,
            indivduals=indivduals,
            probs=all_Pi_Theta_exp,
            nb_instances=B,
            num_agents=M,
        ).detach()

        loss = torch.mean(advantages * log_Pi, dim=1).sum()

        with torch.no_grad():
            self.baseline = fitness.mean(dim=1).view(B, M)
            self.latest_advantages = advantages.view(B, M, λa).reshape(B, self.lambda_).cpu()

        (grad_theta,) = torch.autograd.grad(loss, self.theta, create_graph=False, retain_graph=True)
        self.last_theta_grad = grad_theta.detach().clone()

        if self.debug_svgd:
            with torch.no_grad():
                self._last_debug_stats = {
                    "loss_mean": float(loss.item()),
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

    def _compute_log_pi(self, indivduals):
        """
        Calcule log π_θ(x) pour chaque échantillon sous self.probs (politique courante).
        indivduals : (BM, λa, N)
        Retourne   : log_Pi (BM, λa)
        """
        BM, λa, N = indivduals.shape
        if self.use_categorical:
            D = self.probs.size(-1)
            probs = self.probs.view(BM, N, D)
            probs_exp = probs.unsqueeze(1).expand(-1, λa, -1, -1)       # (BM, λa, N, D)
            log_probs = torch.log(probs_exp + 1e-10)
            indices = indivduals.long().unsqueeze(-1)                    # (BM, λa, N, 1)
            return log_probs.gather(-1, indices).squeeze(-1).sum(dim=2)  # (BM, λa)
        else:
            probs = self.probs.view(BM, N)
            probs_exp = probs.unsqueeze(1).expand(-1, λa, -1)           # (BM, λa, N)
            Pi_sel = torch.where(indivduals == 1.0, probs_exp, 1.0 - probs_exp)
            return torch.log(Pi_sel + 1e-10).sum(dim=2)                 # (BM, λa)

    def _updateDistribution_PPO(self, solutionList, scoreList):
        """
        Implémente SVGD-EDA-PPO :
          θ_old ← θ
          for k = 1..K :
              r_jl = exp(log π_θ - log π_θ_old)
              g_j  = (1/λγ) Σ_l ∇ min(r_jl·Ŝ, clip(r_jl, 1-ε, 1+ε)·Ŝ)
              θ_i += ε · SVGD_phi(θ, g)
        """
        B, M, N = self.nb_instances, self.M, self.N
        λa = self.lambda_per_agent
        BM = B * M

        indivduals = solutionList.view(BM, λa, N)
        fitness = scoreList.view(BM, λa)
        baseline = self.baseline.view(BM) if self.baseline.numel() > 0 else torch.zeros(BM, device=self.device)

        # --- Avantages Ŝ : calculés une fois, fixes pour les K époques ---
        if self.use_categorical:
            D = self.probs.size(-1)
            theta_flat = self.theta.view(BM, N, D)
            all_Pi_Theta = self.probs.view(BM, N, D)
            all_Pi_Theta_exp = all_Pi_Theta.unsqueeze(1).expand(-1, λa, -1, -1)
        else:
            theta_flat = self.theta.view(BM, N)
            all_Pi_Theta = self.probs.view(BM, N)
            all_Pi_Theta_exp = all_Pi_Theta.unsqueeze(1).expand(-1, λa, -1)

        advantages = self.advantage_strategy.compute(
            fitness=fitness,
            baseline=baseline,
            theta=theta_flat,
            indivduals=indivduals,
            probs=all_Pi_Theta_exp,
            nb_instances=B,
            num_agents=M,
        ).detach()  # coefficient fixe, pas de gradient nécessaire

        # --- π_θ_old figées pour les K époques (ratio + KL) ---
        with torch.no_grad():
            pi_old_per_pos = self._compute_pi_per_pos(indivduals).detach()  # (BM, λa, N)
            if self.ppo_mode in ('kl', 'trpo'):
                pi_old_full = self.probs.view(BM, N, -1).detach() if self.use_categorical \
                    else self.probs.view(BM, N).detach()

        # --- Mise à jour de la baseline et stockage des avantages ---
        with torch.no_grad():
            self.baseline = fitness.mean(dim=1).view(B, M)
            self.latest_advantages = advantages.view(B, M, λa).reshape(B, self.lambda_).cpu()

        # --- Boucle K époques PPO avec SVGD intra-boucle ---
        last_surrogate_mean = None
        adv_exp = advantages.unsqueeze(-1)  # (BM, λa, 1) — précalculé hors boucle

        for k in range(self.ppo_epochs):
            # Recalcul de π_θ depuis le theta courant (potentiellement mis à jour par SVGD)
            self.probs = self.forward()

            # Ratio par position : r_n = π_θ_new(x_n) / π_θ_old(x_n)
            pi_new_per_pos = self._compute_pi_per_pos(indivduals)  # (BM, λa, N)
            ratio = pi_new_per_pos / pi_old_per_pos                # (BM, λa, N)
            surr1 = ratio * adv_exp                                # (BM, λa, N)

            if self.ppo_mode == 'clip':
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_exp
                surrogate = torch.min(surr1, surr2).sum(dim=-1)   # (BM, λa)
                objective = surrogate.mean(dim=1).sum()

            else:  # 'kl' | 'trpo'
                surrogate = surr1.sum(dim=-1)                      # (BM, λa) — pas de clipping
                pi_new_full = self.probs.view(BM, N, -1) if self.use_categorical \
                    else self.probs.view(BM, N)
                kl_sum, kl_mean = self._compute_kl(pi_old_full, pi_new_full)

                if self.ppo_mode == 'trpo':
                    # Pas de pénalité -beta*KL : la contrainte est appliquée sur le PAS SVGD lui-même
                    # (backoff façon line-search TRPO), cf. _apply_svgd_with_kl_constraint.
                    objective = surrogate.mean(dim=1).sum()
                else:  # 'kl'
                    objective = surrogate.mean(dim=1).sum() - self.kl_beta * kl_sum

                    # Adaptation de β à la dernière epoch seulement (KL moyenne par position après K steps)
                    if self.kl_target_kl is not None and k == self.ppo_epochs - 1:
                        with torch.no_grad():
                            kl_val = float(kl_mean.detach().item())
                            if kl_val > 1.5 * self.kl_target_kl:
                                self.kl_beta = min(self.kl_beta * 1.5, self.kl_beta_max)
                            elif kl_val < self.kl_target_kl / 1.5:
                                self.kl_beta = max(self.kl_beta / 1.5, self.kl_beta_min)

            # retain_graph=False : le graph est reconstruit à chaque epoch via forward()
            (grad_theta,) = torch.autograd.grad(
                objective, self.theta, create_graph=False, retain_graph=False
            )
            self.last_theta_grad = grad_theta.detach().clone()

            if self.debug_svgd and self._last_debug_stats is not None:
                with torch.no_grad():
                    self._last_debug_stats["theta_grad_norm"] = float(grad_theta.norm().item())

            # Pas SVGD : θ_i += ε · φ(θ)  — à l'intérieur de la boucle K
            if self.ppo_mode == 'trpo':
                self._apply_svgd_with_kl_constraint(pi_old_full, BM, N)
            else:
                self._apply_svgd()

            last_surrogate_mean = float(surrogate.detach().mean(dim=1).mean().item())

        # --- Stats de debug (dernière époque) ---
        if self.debug_svgd:
            with torch.no_grad():
                self.probs = self.forward()
                log_Pi = self._compute_log_pi(indivduals)
                self._last_debug_stats = {
                    "loss_mean": float(last_surrogate_mean) if last_surrogate_mean is not None else float("nan"),
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

        val = last_surrogate_mean if last_surrogate_mean is not None else 0.0
        return torch.tensor(val, device=self.device)

    def get_latest_advantages(self):
        if self.latest_advantages is None:
            return None
        return self.latest_advantages.detach().cpu()

    def toString(self):
        return f"MultiAgent_Collaborative_M{self.M}_lambdaPerAgent{self.lambda_per_agent}"

    def _apply_svgd(self, step_scale=1.0):
        """
        Applique un pas SVGD instance par instance en se basant sur les directions RL observées.
        Utilise self.last_theta_grad comme direction RL : (B, M, N)
        `step_scale` permet de réduire le pas (backoff façon line-search, cf. contrainte KL dure).
        """
        if self.last_theta_grad is None:
            return

        theta = self.theta
        score = self.last_theta_grad.detach()

        with torch.enable_grad():
            # Recompute fresh probs so the kernel always has a valid graph
            # (the PPO/REINFORCE backward may have freed the previous one)
            probs = self.forward()
            phi = self.svgd.phi(theta, score, probs=probs)
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
            self.theta += step_scale * self.epsilon_svgd * phi
            self.probs = None

    def _apply_svgd_with_kl_constraint(self, pi_old_full, BM, N):
        """
        Mode 'trpo' : applique le pas SVGD sous contrainte dure de KL, façon line-search TRPO,
        au lieu d'une pénalité -beta*KL dans l'objectif.

        Le nombre d'époques K (`ppo_epochs`) n'est pas affecté : cette méthode gère uniquement
        CE QUI SE PASSE À L'INTÉRIEUR d'une époque donnée (le pas SVGD de cette époque), la boucle
        `for k in range(self.ppo_epochs)` dans `_updateDistribution_PPO` reste inchangée — on fait
        toujours K époques, chacune avec son propre essai (et éventuel backoff) de pas SVGD.

        Si la KL_mean mesurée APRES le pas dépasse trpo_kl_threshold, on annule le pas et on
        retente avec un pas réduit de moitié (jusqu'à trpo_backoff_max_tries tentatives). Si même
        la plus petite tentative dépasse encore le seuil, on la garde quand même (on n'annule pas
        totalement le pas de cette époque) : sinon, si epsilon_svgd est mal calibré, l'entraînement
        pourrait ne plus jamais bouger pour cette époque (0 mise à jour), ce qui bloquerait tout.
        """
        if self.last_theta_grad is None:
            return

        theta_before = self.theta.detach().clone()
        step_scale = 1.0
        kl_mean_after = None
        for attempt in range(self.trpo_backoff_max_tries):
            self._apply_svgd(step_scale=step_scale)
            with torch.no_grad():
                probs_after = self.forward()
                pi_after_full = probs_after.view(BM, N, -1) if self.use_categorical \
                    else probs_after.view(BM, N)
                _, kl_mean_after = self._compute_kl(pi_old_full, pi_after_full)
            kl_val = float(kl_mean_after)
            ok = kl_val <= self.trpo_kl_threshold
            if ok:
                break
            if attempt < self.trpo_backoff_max_tries - 1:
                self.theta.data.copy_(theta_before)
                self.probs = None
                step_scale *= 0.5

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
        self._apply_gamma_decay((current_iter + 1) / float(total_iters))

    def decay_svgd_gamma_by_budget(self, evals_consumed: int, budget: int) -> None:
        """
        Variante de decay_svgd_gamma basée sur la fraction du budget d'évaluations
        déjà consommée plutôt que sur l'index d'itération. Nécessaire dès que la
        taille de batch varie d'une itération à l'autre (batch adaptatif) : dans ce
        cas, "itération n / nb_iterations" ne correspond plus à une fraction fixe du
        budget, alors que evals_consommés / budget conserve son sens.
        """
        self._apply_gamma_decay(float(evals_consumed) / float(budget) if budget else 0.0)

    def _apply_gamma_decay(self, progress: float) -> None:
        if not self.decay_enabled or self.no_interact or self.decay_start_ratio >= 1.0 or self.decay_min_factor >= 1.0:
            return
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
        if getattr(self, "record_full_theta_history", True):
            self.theta_history.append(probs_final)
        else:
            # Panneau : seul le dernier snapshot (fin de budget) est conservé.
            self.theta_history = [probs_final]

    def get_theta_history(self):
        return {"values": self.theta_history}

    def get_kernel_metric_history(self):
        return list(self.kernel_metric_history)

    def get_latest_kernel_metrics(self):
        if not self.kernel_metric_history:
            return None
        return self.kernel_metric_history[-1]

    def get_latest_force_stats(self):
        return self.svgd.get_last_force_stats()

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
