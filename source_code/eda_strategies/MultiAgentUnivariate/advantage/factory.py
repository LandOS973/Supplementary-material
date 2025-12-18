from typing import Any, Dict, Type

from .base import AdvantageStrategy
from .baseline import BaselineAdvantage
from .rank_weighted import (
    GlobalRankWeightedAdvantage,
    NormalizedFitnessAdvantage,
    PerAgentRankWeightedAdvantage,
)


class AdvantageFactory:
    _REGISTRY: Dict[str, Type[AdvantageStrategy]] = {
        "baseline": BaselineAdvantage,
        "globalrankweighted": GlobalRankWeightedAdvantage,
        "rankbaseweighted": GlobalRankWeightedAdvantage,  # rétrocompatibilité
        "peragentrankweighted": PerAgentRankWeightedAdvantage,
        "normalizedfitness": NormalizedFitnessAdvantage,
    }

    @classmethod
    def register(cls, name: str, strategy_cls: Type[AdvantageStrategy]):
        cls._REGISTRY[name.lower()] = strategy_cls

    @classmethod
    def create(cls, name: str | None, params: Dict[str, Any] | None = None) -> AdvantageStrategy:
        key = (name or "baseline").lower()
        if key not in cls._REGISTRY:
            raise ValueError(f"Unknown advantage strategy '{name}'. Available: {sorted(cls._REGISTRY)}")
        strategy_cls = cls._REGISTRY[key]
        params = params or {}
        return strategy_cls(**params)

    @classmethod
    def from_config(cls, config: Dict[str, Any] | str | None) -> AdvantageStrategy:
        if config is None:
            config_dict: Dict[str, Any] = {}
        elif isinstance(config, str):
            config_dict = {"type": config}
        elif isinstance(config, dict):
            config_dict = config
        else:
            raise TypeError(f"Unsupported advantage config type: {type(config)}")
        name = config_dict.get("type", "baseline")
        params = config_dict.get("params", {})
        return cls.create(name, params)
