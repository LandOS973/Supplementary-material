import torch

from .base import AdvantageStrategy


class BaselineAdvantage(AdvantageStrategy):
    """
    Avantage classique: reward - baseline (fitness moyen de l'iteration precedente).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._baseline = None

    def compute(self, fitness, **context):
        if self._baseline is None or self._baseline.shape[0] != fitness.shape[0]:
            baseline = torch.zeros(fitness.shape[0], device=fitness.device, dtype=fitness.dtype)
        else:
            baseline = self._baseline.to(device=fitness.device, dtype=fitness.dtype)
        advantage = fitness - baseline.unsqueeze(1)
        with torch.no_grad():
            self._baseline = fitness.mean(dim=1).detach().clone()
        return advantage
