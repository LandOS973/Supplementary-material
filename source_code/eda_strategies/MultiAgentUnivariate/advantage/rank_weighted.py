from .base import AdvantageStrategy


class RankBaseWeightedAdvantage(AdvantageStrategy):
    """
    Placeholder : transformation basée sur le rang pondéré.
    Implémentation future.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.message = kwargs.get("message", "RankBaseWeightedAdvantage not implemented yet.")

    def compute(self, fitness, baseline, **context):
        raise NotImplementedError(self.message)
