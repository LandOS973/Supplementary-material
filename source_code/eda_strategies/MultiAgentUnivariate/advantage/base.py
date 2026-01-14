from abc import ABC, abstractmethod


class AdvantageStrategy(ABC):
    """
    Interface commune pour toutes les variantes d'estimation d'avantage.
    """

    def __init__(self, **kwargs):
        self.extra_kwargs = kwargs

    @abstractmethod
    def compute(self, fitness, baseline, **context):
        """
        Args:
            fitness: Tensor (BM, λa)
            baseline: Tensor (BM,)
            context: Informations optionnelles (theta, indivduals, etc.)
        Returns:
            Tensor (BM, λa)
        """
        raise NotImplementedError
