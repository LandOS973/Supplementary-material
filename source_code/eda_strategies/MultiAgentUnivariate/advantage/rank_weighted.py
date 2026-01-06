import torch

from .base import AdvantageStrategy


class GlobalRankWeightedAdvantage(AdvantageStrategy):
    """
    Classement pondéré global :
    - On rassemble tous les individus d'une instance (tous agents confondus)
    - On les trie par fitness décroissante
    - On attribue des poids linéaires de start_weight à end_weight en fonction du rang global
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.start_weight = float(1.0)   # w_max
        self.end_weight = float(-1.0)    # w_min

    def compute(self, fitness, nb_instances=None, num_agents=None, **context):
        nb_instances = nb_instances or context.get("nb_instances")
        num_agents = num_agents or context.get("num_agents")
        nb_instances = int(nb_instances)
        num_agents = int(num_agents)

        BM, lambda_per_agent = fitness.shape  # BM = B * M
        # reshape -> (B, M, λ_agent)
        reshaped = fitness.view(nb_instances, num_agents, lambda_per_agent)
        # (B, M * λ_agent) : tous les individus d'une instance sur une seule dimension
        per_instance = reshaped.view(nb_instances, -1)

        # Nombre d'individus par instance : λ = M * λ_agent
        num_individuals = per_instance.shape[1]
        # Pas positif Δ = (w_max - w_min) / (λ - 1)
        delta = (self.start_weight - self.end_weight) / (num_individuals - 1)

        # Rang r = 0,...,λ-1 → w_r = w_max - r * Δ
        ranks = torch.arange(
            num_individuals,
            device=fitness.device,
            dtype=fitness.dtype,
        )
        weight_vector = self.start_weight - ranks * delta  # (λ,)

        # On duplique ce vecteur pour toutes les instances : (B, λ)
        weights = weight_vector.unsqueeze(0).expand_as(per_instance)

        # Classement global par instance (descendant)
        sorted_indices = torch.argsort(per_instance, dim=1, descending=True)

        # On scatter les poids selon le rang : les meilleurs reçoivent les plus grands poids
        ranked = torch.empty_like(per_instance).scatter_(1, sorted_indices, weights)

        # Retour au shape original (B*M, λ_agent)
        return ranked.view(BM, lambda_per_agent)


class PerAgentRankWeightedAdvantage(AdvantageStrategy):
    """Classement pondéré indépendant par agent.

    Pour chaque agent m :
    - on trie ses λ_agent individus par fitness décroissante
    - on attribue des poids linéaires de start_weight à end_weight
      en fonction du rang au sein de l'agent
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.start_weight = float(1.0)   # w_max
        self.end_weight = float(-1.0)    # w_min

    def compute(self, fitness, nb_instances=None, num_agents=None, **context):
        nb_instances = nb_instances or context.get("nb_instances")
        num_agents = num_agents or context.get("num_agents")
        nb_instances = int(nb_instances)
        num_agents = int(num_agents)

        BM, lambda_per_agent = fitness.shape  # BM = B * M

        # reshape -> (B, M, λ_agent)
        reshaped = fitness.view(nb_instances, num_agents, lambda_per_agent)

        # Pas positif Δ = (w_max - w_min) / (λ_agent - 1)
        delta = (self.start_weight - self.end_weight) / (lambda_per_agent - 1)

        # Rang r = 0,...,λ_agent-1 → v_r = w_max - r * Δ
        ranks = torch.arange(
            lambda_per_agent,
            device=fitness.device,
            dtype=fitness.dtype,
        )
        base_weights = self.start_weight - ranks * delta  # (λ_agent,)

        # On étend à (B, M, λ_agent)
        weight_vector = base_weights.view(1, 1, -1).expand_as(reshaped)

        # Tri par agent sur la dimension λ_agent (dim=2)
        sorted_indices = torch.argsort(reshaped, dim=2, descending=True)

        # On scatter les poids selon le rang intra-agent
        ranked = torch.empty_like(reshaped).scatter_(2, sorted_indices, weight_vector)

        # Retour au shape original (B*M, λ_agent)
        return ranked.view(BM, lambda_per_agent)


class NormalizedFitnessAdvantage(AdvantageStrategy):
    """
    Normalise la fitness de chaque individu j de l'agent i à l'itération t selon :
        h_{i,j,t} = (f(x_{i,j,t}) - f^{ref}_{i,t}) / (f^{max}_t - f^{ref}_{i,t})

    où f^{ref}_{i,t} est la fitness moyenne de l'agent i à l'itération précédente
    (fournie via `baseline`) et f^{max}_t est la meilleure fitness observée jusqu'ici
    pour l'instance correspondante. Résultat borné automatiquement via eps.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_fitness = None  # (B,)
        self.fitness_mean = None  # (B,)


    def compute(self, fitness, nb_instances=None, num_agents=None, **context):
        nb_instances = nb_instances or context.get("nb_instances")
        num_agents = num_agents or context.get("num_agents")
        if nb_instances is None or num_agents is None:
            raise ValueError("nb_instances and num_agents must be provided for NormalizedFitnessAdvantage.")
        nb_instances = int(nb_instances)
        num_agents = int(num_agents)

        BM, lambda_per_agent = fitness.shape
        device, dtype = fitness.device, fitness.dtype

        current_fitness = fitness.view(nb_instances, num_agents, lambda_per_agent)

        if self.fitness_mean is None or self.fitness_mean.shape[0] != nb_instances:
            previous_mean = torch.zeros(nb_instances, device=device, dtype=dtype)
        else:
            previous_mean = self.fitness_mean.to(device=device, dtype=dtype)
        instance_baseline = previous_mean.view(nb_instances, 1, 1)

        current_best = current_fitness.view(nb_instances, -1).max(dim=1).values
        if self.best_fitness is None:
            self.best_fitness = current_best.detach().clone()
        else:
            self.best_fitness = torch.maximum(self.best_fitness, current_best)
        best_per_instance = self.best_fitness.view(nb_instances, 1, 1)

        denom = best_per_instance - instance_baseline
        eps = torch.finfo(dtype).eps
        safe_denom = torch.where(torch.abs(denom) < eps, torch.ones_like(denom), denom)
        normalized = (current_fitness - instance_baseline) / safe_denom
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

        with torch.no_grad():
            self.fitness_mean = current_fitness.view(nb_instances, -1).mean(dim=1).detach().clone()

        return normalized.view(BM, lambda_per_agent)
