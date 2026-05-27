import csv
import os
from pathlib import Path

import torch

from .base import AdvantageStrategy


class GlobalRankWeightedAdvantage(AdvantageStrategy):
    """
    Classement pondéré global (version IGO/RL-EDA) :
    - On rassemble tous les individus d'une instance (tous agents confondus)
    - On les trie par fitness décroissante
    - Les ex-aequo partagent le même rang (0, 0, 2, ...)
    - On applique w(x)=1-2x avec x = rangGlobal / (M * λ_agent)
      => poids = 1 - 2 * (rangGlobal / (M * λ_agent))
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute(self, fitness, nb_instances=None, num_agents=None, mask=None, **context):
        nb_instances = nb_instances or context.get("nb_instances")
        num_agents = num_agents or context.get("num_agents")
        nb_instances = int(nb_instances)
        num_agents = int(num_agents)

        BM, lambda_per_agent = fitness.shape
        per_instance = fitness.view(nb_instances, -1)

        if mask is not None:
            # Remplace les faux samples par -inf : ils se retrouvent derniers dans le
            # classement et n'affectent pas le rang des vrais samples.
            # La somme réelle est conservée (M × λ_init), donc total_individuals est
            # le même pour toutes les instances.
            mask_2d = mask.view(nb_instances, -1)
            ranking_fitness = per_instance.masked_fill(mask_2d == 0, float("-inf"))
            total_individuals = int(mask_2d[0].sum().item())
        else:
            ranking_fitness = per_instance
            total_individuals = lambda_per_agent * num_agents

        greater_counts = (ranking_fitness[:, :, None] < ranking_fitness[:, None, :]).sum(dim=2)
        ranks = greater_counts.to(dtype=fitness.dtype)
        ranked = 1.0 - 2.0 * (ranks / total_individuals)
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
        self.start_weight = float(1.0)          
        self.end_weight = float(-1.0)           

    def compute(self, fitness, nb_instances=None, num_agents=None, **context):
        nb_instances = nb_instances or context.get("nb_instances")
        num_agents = num_agents or context.get("num_agents")
        nb_instances = int(nb_instances)
        num_agents = int(num_agents)

        BM, lambda_per_agent = fitness.shape              

        reshaped = fitness.view(nb_instances, num_agents, lambda_per_agent)

        if lambda_per_agent == 1:
            base_weights = torch.full(
                (1,),
                self.start_weight,
                device=fitness.device,
                dtype=fitness.dtype,
            )
        else:
            delta = (self.start_weight - self.end_weight) / (lambda_per_agent - 1)
            ranks = torch.arange(
                lambda_per_agent,
                device=fitness.device,
                dtype=fitness.dtype,
            )
            base_weights = self.start_weight - ranks * delta              

        weight_vector = base_weights.view(1, 1, -1).expand_as(reshaped)

        sorted_indices = torch.argsort(reshaped, dim=2, descending=True)

        ranked = torch.empty_like(reshaped).scatter_(2, sorted_indices, weight_vector)

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
        self.best_fitness = None        
        self.fitness_mean = None        


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


class BaselineRescaledAdvantage(AdvantageStrategy):
    """
    Avantage basé sur une calibration fixe : f - b.
    b est calculé une seule fois à partir d'un fichier de calibration
    par distribution d'instances, puis mis en cache.
    """

    def __init__(
        self,
        calibration_path: str | None = None,
        problem: str | None = None,
        dim: int | None = None,
        type_instance: int | None = None,
        top_k: int = 200,
        h_top_k: int = 500,
    ):
        super().__init__()
        self.calibration_path = calibration_path
        self.problem = (problem or "").upper()
        self.dim = dim
        self.type_instance = type_instance
        self.top_k = int(top_k)
        self.h_top_k = int(h_top_k)
        self._cached_b = None

    def _find_repo_root(self) -> Path:
        here = Path(__file__).resolve()
        for parent in [here] + list(here.parents):
            if (parent / "additional_results").is_dir() and (parent / "config").is_dir():
                return parent
        return Path.cwd()

    def _infer_problem_cfg(self, repo_root: Path):
        cfg_path = repo_root / "config" / "config.yaml"
        if not cfg_path.exists():
            return None
        try:
            with cfg_path.open("r", encoding="utf-8") as fh:
                lines = fh.readlines()
        except OSError:
            return None
        problem_key = None
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("- problem:"):
                problem_key = stripped.split(":", 1)[1].strip()
                if "#" in problem_key:
                    problem_key = problem_key.split("#", 1)[0].strip()
                break
        if not problem_key:
            return None
        problem_cfg = repo_root / "config" / "problem" / f"{problem_key}.yaml"
        if not problem_cfg.exists():
            return None
        data = {}
        try:
            with problem_cfg.open("r", encoding="utf-8") as fh:
                for raw in fh:
                    row = raw.strip()
                    if not row or row.startswith("#") or ":" not in row:
                        continue
                    key, value = row.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    if "#" in value:
                        value = value.split("#", 1)[0].strip()
                    if not value:
                        continue
                    lowered = value.lower()
                    if lowered in ("none", "null"):
                        parsed = None
                    else:
                        try:
                            parsed = int(value)
                        except ValueError:
                            parsed = value
                    data[key] = parsed
        except OSError:
            return None
        return data

    def _resolve_calibration_path(self) -> Path:
        if self.calibration_path:
            return Path(self.calibration_path)

        repo_root = self._find_repo_root()
        self._ensure_problem_meta(repo_root)
        problem = self.problem
        dim = self.dim
        type_instance = self.type_instance

        if not problem or dim is None or type_instance is None:
            raise ValueError("Impossible de déterminer problem/dim/type_instance pour baseline_rescaled.")

        if problem in ("QUBO", "UBQP"):
            filename = f"UBQP_N_{dim}_K_{type_instance}_ranks.csv"
        elif problem == "NK3":
            filename = f"NK3_N_{dim}_K_{type_instance}_ranks.csv"
        elif problem == "NK":
            filename = f"NK_N_{dim}_K_{type_instance}_ranks.csv"
        else:
            raise ValueError(f"Problem {problem} non supporté pour baseline_rescaled.")

        return repo_root / "additional_results" / "global_ranking" / filename

    def _ensure_problem_meta(self, repo_root: Path | None = None):
        if self.problem and self.dim is not None and self.type_instance is not None:
            return
        repo_root = repo_root or self._find_repo_root()
        cfg = self._infer_problem_cfg(repo_root) or {}
        if not self.problem:
            self.problem = str(cfg.get("name") or cfg.get("type_problem") or "").upper()
        if self.dim is None:
            self.dim = cfg.get("dim") or cfg.get("n")
        if self.type_instance is None:
            self.type_instance = cfg.get("type_instance") or cfg.get("k")

    def _load_scores(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Calibration introuvable: {path}")
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            rows = [row for row in reader if row]
        if not rows:
            raise ValueError(f"Calibration vide: {path}")
        header = [h.strip().lower() for h in rows[0]]
        rows = rows[1:]
        idx_score = None
        for key in ("score", "best_score", "value", "objective", "obj"):
            if key in header:
                idx_score = header.index(key)
                break
        if idx_score is None:
            raise ValueError(f"Colonne score introuvable dans {path}")
        scores = []
        for r in rows:
            if idx_score >= len(r):
                continue
            try:
                s = float(r[idx_score])
            except Exception:
                continue
            scores.append(s)
        if not scores:
            raise ValueError(f"Aucun score valide dans {path}")
        return scores

    def _compute_b(self):
        path = self._resolve_calibration_path()
        scores = self._load_scores(path)
        scores_sorted = sorted(scores, reverse=True)
        top_k = min(self.top_k, len(scores_sorted))
        top_scores = scores_sorted[:top_k]
        b = sum(top_scores) / max(1, len(top_scores))
        return b

    def compute(self, fitness, **context):
        if self._cached_b is None:
            self._cached_b = self._compute_b()
        b = self._cached_b
        self._ensure_problem_meta()
        f = fitness
        if self.dim:
            f = f / float(self.dim)
        device, dtype = fitness.device, fitness.dtype
        b_t = torch.tensor(b, device=device, dtype=dtype)
        print("FITNESS DES 5 PREMIER ", f[:5])
        return f - b_t
