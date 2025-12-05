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
        self.start_weight = float(100.0)   # w_max
        self.end_weight = float(-100.0)    # w_min

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
        self.start_weight = float(100.0)   # w_max
        self.end_weight = float(-100.0)    # w_min

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
