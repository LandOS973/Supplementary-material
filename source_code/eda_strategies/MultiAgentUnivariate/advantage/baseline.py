import torch

from .base import AdvantageStrategy


class BaselineAdvantage(AdvantageStrategy):
    """
    Avantage classique: reward - baseline avec option de centrage local.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
   

    def compute(self, fitness, baseline, **context):
        return fitness - baseline.unsqueeze(1)
