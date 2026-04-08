#!/usr/bin/env python3
"""
Hydra runner for MaxCut with SVGD_EDA (PPO-EDA strategy).
"""

from __future__ import annotations

import os
import random
import urllib.request
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from eda_strategies.FactoryStrategyEA import FactoryStrategyEA
from environment.metrics import MetricsCalculator
from environment.tsne_agents import plot_agents_tsne
from environment.visualization import render_agent_dashboard

try:
    import nevergrad as ng
except Exception:  # pragma: no cover
    ng = None


def _load_kernel_config(kernel_name: str, repo_root: str) -> dict:
    kernel_dir = Path(repo_root) / "config" / "kernel"
    kernel_path = kernel_dir / f"{kernel_name}.yaml"
    if not kernel_path.exists():
        available = ", ".join(sorted(p.stem for p in kernel_dir.glob("*.yaml"))) if kernel_dir.exists() else "none"
        raise FileNotFoundError(
            f"Kernel config '{kernel_name}' introuvable dans {kernel_dir}. Kernels disponibles: {available}"
        )
    cfg = OmegaConf.load(str(kernel_path))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True) or {}
    if "name" not in cfg_dict:
        cfg_dict["name"] = kernel_name
    return cfg_dict


def load_gset_matrix(instance_name: str, url_base: str, device: str) -> tuple[torch.Tensor, int, int]:
    """
    Load one Gset-like MaxCut instance as a symmetric adjacency matrix.
    Expected format:
      line 1: N E
      next lines: u v weight
    Node ids in files are 1-based and converted to 0-based.
    """
    script_dir = Path(__file__).resolve().parent
    cache_dir = script_dir / "instances" / "maxcut"
    cache_dir.mkdir(parents=True, exist_ok=True)

    requested = Path(instance_name)
    local_candidates = []
    if requested.is_absolute():
        local_candidates.append(requested)
    else:
        local_candidates.append(cache_dir / requested.name)
        local_candidates.append(script_dir / requested)
        local_candidates.append(Path.cwd() / requested)

    local_path = next((path for path in local_candidates if path.exists()), None)

    if local_path is None:
        if not url_base:
            raise FileNotFoundError(
                f"Instance '{instance_name}' introuvable localement et url_base vide, impossible de telecharger."
            )
        url = f"{url_base.rstrip('/')}/{instance_name}"
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                raw_content = response.read()
        except Exception as exc:
            raise RuntimeError(f"Echec du telechargement de '{url}': {exc}") from exc
        text = raw_content.decode("utf-8", errors="strict")
        local_path = cache_dir / requested.name
        local_path.write_text(text, encoding="utf-8")
        print(f"[INFO] downloaded {instance_name} -> {local_path}")
    else:
        text = local_path.read_text(encoding="utf-8")
        print(f"[INFO] using local MaxCut instance: {local_path}")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Fichier instance vide: {local_path}")

    header = lines[0].split()
    if len(header) < 2:
        raise ValueError(f"Entete invalide dans {local_path}: '{lines[0]}' (attendu: N E)")

    n = int(header[0])
    declared_edges = int(header[1])
    adjacency = torch.zeros((n, n), dtype=torch.float32, device=device)
    parsed_edges = 0

    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        u = int(parts[0]) - 1
        v = int(parts[1]) - 1
        w = float(parts[2])
        if u < 0 or v < 0 or u >= n or v >= n:
            raise ValueError(f"Index hors bornes dans {local_path}: '{line}'")
        adjacency[u, v] += w
        adjacency[v, u] += w
        parsed_edges += 1

    adjacency.fill_diagonal_(0.0)
    num_edges = parsed_edges if parsed_edges > 0 else declared_edges
    return adjacency, n, num_edges


def evaluate_maxcut_batch(agents_batch: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute MaxCut score for a batch of binary solutions.
    agents_batch: (B, N) with values in {0, 1}
    adjacency_matrix: (N, N), symmetric
    returns: (B,)
    """
    if agents_batch.dim() != 2:
        raise ValueError(f"agents_batch must be 2D (B, N), got shape={tuple(agents_batch.shape)}")
    x = agents_batch.to(dtype=adjacency_matrix.dtype)
    x_bar = 1.0 - x
    ax = torch.matmul(x, adjacency_matrix)
    return torch.sum(ax * x_bar, dim=1)


def _run_nevergrad_maxcut(
    adjacency_matrix: torch.Tensor,
    budget: int,
    algo_name: str,
    seed: int,
    verbose: bool,
) -> tuple[float, torch.Tensor]:
    if ng is None:
        raise RuntimeError(
            "Import `nevergrad` failed. Install Nevergrad first (e.g. `pip install nevergrad`)."
        )

    n = int(adjacency_matrix.shape[0])
    parametrization = ng.p.TransitionChoice(range(2), repetitions=n, ordered=False)
    algo_cls = ng.optimizers.registry.get(algo_name)
    if algo_cls is None:
        available = sorted(ng.optimizers.registry.keys())
        raise ValueError(f"Unknown Nevergrad algo '{algo_name}'. Available: {available}")
    optimizer = algo_cls(parametrization=parametrization, budget=budget)
    optimizer.parametrization.random_state.seed(seed)

    best_score = float("-inf")
    best_solution = None
    iterator = range(1, budget + 1)
    progress = (
        tqdm(iterator, total=budget, desc=f"Nevergrad {algo_name}", dynamic_ncols=True, disable=not verbose)
        if tqdm is not None
        else iterator
    )

    for _ in progress:
        candidate = optimizer.ask()
        x_np = np.asarray(candidate.value, dtype=np.float32)
        x = torch.from_numpy(np.rint(x_np)).to(adjacency_matrix.device).unsqueeze(0)
        score = float(evaluate_maxcut_batch(x, adjacency_matrix).item())
        optimizer.tell(candidate, -score)

        if score > best_score:
            best_score = score
            best_solution = x.squeeze(0).detach().clone()

        if tqdm is not None and verbose:
            progress.set_postfix(best_global=f"{best_score:.2f}")

    if best_solution is None:
        raise RuntimeError("Nevergrad did not produce any candidate.")
    return best_score, best_solution


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"running on device: {device}")

    script_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    maxcut_cfg_path = Path(repo_root) / "config" / "problem" / "maxcut.yaml"
    maxcut_defaults = OmegaConf.load(str(maxcut_cfg_path)) if maxcut_cfg_path.exists() else OmegaConf.create({})

    def agent_val(key):
        try:
            return OmegaConf.select(cfg, f"agent.{key}")
        except Exception:
            return None

    problem_name = str(cfg.problem.name if "problem" in cfg and "name" in cfg.problem else "MAXCUT").upper()
    instance_name = (
        OmegaConf.select(cfg, "problem.instance_name")
        or cfg.get("instance_name")
        or OmegaConf.select(maxcut_defaults, "instance_name")
    )
    url_base = (
        OmegaConf.select(cfg, "problem.url_base")
        or cfg.get("url_base")
        or OmegaConf.select(maxcut_defaults, "url_base")
        or "https://web.stanford.edu/~yyye/yyye/Gset"
    )
    problem_nevergrad = OmegaConf.select(cfg, "problem.nevergrad")
    if problem_nevergrad is None:
        problem_nevergrad = OmegaConf.select(maxcut_defaults, "nevergrad")
    use_nevergrad = bool(problem_nevergrad if problem_nevergrad is not None else cfg.get("nevergrad", False))
    nevergrad_algo = str(
        OmegaConf.select(cfg, "problem.algo_name")
        or OmegaConf.select(maxcut_defaults, "algo_name")
        or cfg.get("nevergrad_algo")
        or "DiscreteDE"
    )

    if not instance_name:
        raise ValueError("No MaxCut instance_name found. Set problem.instance_name (e.g. G70).")
    instance_name = str(instance_name)
    url_base = str(url_base)

    if problem_name != "MAXCUT" and not use_nevergrad:
        print(f"[WARN] problem.name={problem_name} (attendu MAXCUT). Le script continue en mode MaxCut.")

    adjacency_matrix, N, E = load_gset_matrix(instance_name, url_base, device)
    print(f"[INFO] graph stats: nodes={N} edges={E}")

    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    lambda_ = int(agent_val("lambda") or cfg.get("lambda") or cfg.get("lambda_") or 10)
    M = int(agent_val("M") or cfg.get("M") or 1)
    verbose = bool(cfg.get("verbose", True))
    budget = int(cfg.get("budget", 10000))
    visualization_enabled = bool(cfg.get("visualization", True))
    advantage_cfg = agent_val("advantage") or cfg.get("advantage") or "baseline"
    if isinstance(advantage_cfg, DictConfig):
        advantage_cfg = OmegaConf.to_container(advantage_cfg, resolve=True)
    no_interact = bool(agent_val("no_interact") or cfg.get("no_interact") or False)
    no_repulsion = bool(agent_val("no_repulsion") or cfg.get("no_repulsion") or False)
    decay_enabled = bool(agent_val("decay") or cfg.get("decay") or False)
    enable_greedy_final = agent_val("enable_greedy_final")
    if enable_greedy_final is None:
        enable_greedy_final = cfg.get("enable_greedy_final", True)
    enable_greedy_final = bool(enable_greedy_final)
    kernel_name = str(agent_val("kernel") or cfg.get("kernel") or "hk").lower()
    kernel_cfg = _load_kernel_config(kernel_name, repo_root)
    prob_eps_override = agent_val("prob_eps_clamp") or cfg.get("prob_eps_clamp")
    if prob_eps_override is not None:
        kernel_cfg["prob_eps_clamp"] = float(prob_eps_override)
    natural_grad_override = agent_val("natural_grad") or cfg.get("natural_grad")
    if natural_grad_override is not None:
        kernel_cfg["natural_grad"] = bool(natural_grad_override)
    bandwith_override = agent_val("bandwith_kernel") or cfg.get("bandwith_kernel")
    if bandwith_override is not None:
        kernel_cfg["bandwith_kernel"] = bandwith_override
    debug_svgd_override = agent_val("debug_svgd")
    if debug_svgd_override is None:
        debug_svgd_override = cfg.get("debug_svgd", False)
    kernel_cfg["debug_svgd"] = bool(debug_svgd_override)

    kernel_lr = kernel_cfg.get("epsilon_svgd")
    kernel_gamma = kernel_cfg.get("gamma")
    epsilon_svgd = float(agent_val("epsilon_svgd") or cfg.get("epsilon_svgd") or kernel_lr or 0.5)
    svgd_gamma = float(agent_val("gamma") or cfg.get("gamma") or kernel_gamma or 10.0)

    decay_default_start_ratio = 0.0 if decay_enabled else 0.8
    decay_default_min_factor = 0.05 if decay_enabled else 0.1
    decay_start_ratio = float(agent_val("decay_start_ratio") or cfg.get("decay_start_ratio") or decay_default_start_ratio)
    decay_min_factor = float(
        agent_val("min_factor")
        or agent_val("decay_min_factor")
        or cfg.get("min_factor")
        or cfg.get("decay_min_factor")
        or decay_default_min_factor
    )

    if use_nevergrad:
        print(
            f"Config: instance={instance_name} N={N} mode=nevergrad | "
            f"algo={nevergrad_algo} budget={budget}"
        )
    else:
        print(
            f"Config: instance={instance_name} N={N} mode=svgd_eda | "
            f"M={M} lambda={lambda_} eps={epsilon_svgd} gamma={svgd_gamma} | "
            f"kernel={kernel_name} advantage={advantage_cfg} decay={decay_enabled} "
            f"greedy_final={enable_greedy_final}"
        )
        if decay_enabled:
            print(f"Decay params: start_ratio={decay_start_ratio} min_factor={decay_min_factor}")

    if use_nevergrad:
        best_score_global, _ = _run_nevergrad_maxcut(
            adjacency_matrix=adjacency_matrix,
            budget=budget,
            algo_name=nevergrad_algo,
            seed=seed,
            verbose=verbose,
        )
        print(f"Instance: {instance_name} | Best Score: {best_score_global}")
        return

    factory = FactoryStrategyEA()
    strategy = factory.createStrategyEA(
        "PPO-EDA",
        N,
        lambda_,
        device,
        dim_variables=None,
        M=M,
        learning_rate=epsilon_svgd,
        epsilon_svgd=epsilon_svgd,
        enable_visualization=visualization_enabled,
        svgd_gamma=svgd_gamma,
        decay_start_ratio=decay_start_ratio,
        decay_min_factor=decay_min_factor,
        decay_enabled=decay_enabled,
        advantage_cfg=advantage_cfg,
        kernel_config=kernel_cfg,
        no_interact=no_interact,
        no_repulsion=no_repulsion,
        is_nk3=False,
    ).to(device)

    if not enable_greedy_final:
        strategy.sample_greedy_agent_solutions = None

    strategy.reset_learned_parameters(nb_instances=1)

    pop_size = M * lambda_
    if pop_size <= 0:
        raise ValueError(f"M*lambda must be > 0 (got M={M}, lambda={lambda_})")

    num_iters = budget // pop_size
    if num_iters <= 0:
        raise ValueError(f"budget={budget} is too small for one iteration with M*lambda={pop_size}")

    best_score_global = float("-inf")
    best_solution_global = None
    collect_dashboard = bool(visualization_enabled) and hasattr(strategy, "agents")
    metrics = MetricsCalculator() if collect_dashboard else None
    iterations = []
    avg_hamming_history = []
    avg_js_history = []
    hamming_pairwise_history = []
    js_pairwise_history = []
    agent_fitness_history = []
    avg_kernel_value_history = []
    avg_kernel_grad_history = []

    iter_range = range(num_iters)
    progress = (
        tqdm(iter_range, total=num_iters, desc=f"MaxCut {instance_name}", dynamic_ncols=True, disable=not verbose)
        if tqdm is not None
        else iter_range
    )

    for it in progress:
        if hasattr(strategy, "decay_svgd_gamma"):
            strategy.decay_svgd_gamma(it, num_iters)

        solutions = strategy.sample_solutions()  # (1, M*lambda, N, 1)
        sols_flat = solutions.view(-1, N)  # (M*lambda, N)
        scores = evaluate_maxcut_batch(sols_flat, adjacency_matrix)  # (M*lambda,)
        strategy.updateDistribution(solutions, scores.view(1, -1))

        best_score_iter = float(scores.max().item())
        if best_score_iter > best_score_global:
            best_idx = int(torch.argmax(scores).item())
            best_score_global = best_score_iter
            best_solution_global = sols_flat[best_idx].detach().clone()

        if tqdm is not None and verbose:
            progress.set_postfix(best_iter=f"{best_score_iter:.2f}", best_global=f"{best_score_global:.2f}")

        if collect_dashboard and metrics is not None:
            agent_mean_scores = scores.view(M, lambda_).mean(dim=1)
            agent_fitness_history.append([float(score.item()) for score in agent_mean_scores])

            avg_hamming, pairwise_hamming = metrics.compute_average_hamming(strategy.agents)
            avg_js, pairwise_js = metrics.compute_average_js(strategy.agents)
            avg_hamming_history.append(avg_hamming if avg_hamming is not None else 0.0)
            avg_js_history.append(avg_js if avg_js is not None else 0.0)
            hamming_pairwise_history.append(pairwise_hamming.tolist() if pairwise_hamming is not None else None)
            js_pairwise_history.append(pairwise_js.tolist() if pairwise_js is not None else None)
            iterations.append((it + 1) * pop_size)

            kernel_stats_fn = getattr(strategy, "get_latest_kernel_metrics", None)
            kernel_stats = kernel_stats_fn() if callable(kernel_stats_fn) else None
            if kernel_stats:
                avg_kernel_value_history.append(kernel_stats.get("avg_kernel_value", 0.0))
                avg_kernel_grad_history.append(kernel_stats.get("avg_kernel_grad", 0.0))
            else:
                avg_kernel_value_history.append(0.0)
                avg_kernel_grad_history.append(0.0)

    if best_solution_global is None:
        raise RuntimeError("No solution sampled during optimization.")

    print(f"Instance: {instance_name} | Best Score: {best_score_global}")

    if collect_dashboard:
        theta_history = None
        theta_history_fn = getattr(strategy, "get_theta_history", None)
        if callable(theta_history_fn):
            theta_history = theta_history_fn()
        render_agent_dashboard(
            iterations=iterations,
            hamming_history=avg_hamming_history,
            js_history=avg_js_history,
            agent_fitness_history=agent_fitness_history,
            num_agents=len(strategy.agents),
            theta_history=theta_history,
            hamming_pairwise_history=hamming_pairwise_history,
            js_pairwise_history=js_pairwise_history,
            kernel_value_history=avg_kernel_value_history,
            kernel_grad_history=avg_kernel_grad_history,
        )

    try:
        plot_agents_tsne(
            strategy,
            output_path=os.path.join(os.getcwd(), "agents_tsne.png"),
            perplexity=None,
            random_state=0,
        )
    except ValueError as exc:
        print(f"[WARN] t-SNE agents skipped: {exc}")


if __name__ == "__main__":
    main()
