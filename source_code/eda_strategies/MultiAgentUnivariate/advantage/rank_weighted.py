import torch

from .base import AdvantageStrategy


class GlobalRankWeightedAdvantage(AdvantageStrategy):
    """
    Classement global strictement basé sur la fitness brute.
    Les poids sont distribués linéairement entre +1 et -1 par instance.
    """
 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.start_weight = float(1.0)
        self.end_weight = float(-1.0)

    def compute(self, fitness, nb_instances=None, num_agents=None, **context):
        nb_instances = nb_instances or context.get("nb_instances")
        num_agents = num_agents or context.get("num_agents")
        nb_instances = int(nb_instances)
        num_agents = int(num_agents)
        BM, lambda_per_agent = fitness.shape

        reshaped = fitness.view(nb_instances, num_agents, lambda_per_agent)
        per_instance = reshaped.view(nb_instances, -1)
        weight_vector = torch.linspace(
            self.start_weight,
            self.end_weight,
            steps=per_instance.shape[1],
            device=fitness.device,
            dtype=fitness.dtype,
        )
        weight_vector_rescaled = weight_vector * lambda_per_agent
        weights = weight_vector_rescaled.unsqueeze(0).expand_as(per_instance)
        sorted_indices = torch.argsort(per_instance, dim=1, descending=True)
        ranked = torch.empty_like(per_instance)
        ranked.scatter_(1, sorted_indices, weights)

        return ranked.view(BM, lambda_per_agent)
