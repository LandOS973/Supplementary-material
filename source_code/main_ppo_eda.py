import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os
import random
from pathlib import Path
from eda_strategies.FactoryStrategyEA import FactoryStrategyEA
from environment.qubo import getTensorInstances_QUBO, get_Score_trajectoriesQUBO_cuda
from environment.blockwise import get_Score_trajectoriesBLOCK_cuda
from environment.nk import getTensorInstances_NK, get_Score_trajectoriesNK_cuda


import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True, formatter={"float_kind": lambda x: f"{x:.6f}"})

# Replication code for the article "Black-Box Combinatorial Optimization with Order-Invariant Reinforcement Learning"


def _load_kernel_config(kernel_name: str, repo_root: str):
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
        # propagate the requested kernel name so downstream components don't fall back to HK
        cfg_dict["name"] = kernel_name
    return cfg_dict


def _parse_summary_config(summary_path):
    cfg = {}
    try:
        with open(summary_path, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                if not value:
                    continue
                lowered = value.lower()
                if lowered in ("none", "null", "n/a"):
                    parsed = None
                elif lowered in ("true", "false"):
                    parsed = lowered == "true"
                else:
                    try:
                        if "." in value:
                            parsed = float(value)
                        else:
                            parsed = int(value)
                    except ValueError:
                        parsed = value
                cfg[key] = parsed
    except OSError:
        return None
    return cfg


def _ask_yes_no(prompt, default=False):
    suffix = " [Y/n]: " if default else " [y/N]: "
    answer = input(prompt + suffix).strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes", "o", "oui")


def _find_best_kernel_summary(summary_dir, type_problem):
    best_kernel = None
    best_cfg = None
    best_score = None
    maximize = type_problem in ("NK", "BLOCK")
    try:
        summary_files = list(Path(summary_dir).glob(f"{type_problem}_*_best_summary.txt"))
    except OSError:
        summary_files = []
    for path in summary_files:
        cfg = _parse_summary_config(path)
        if not cfg:
            continue
        kernel = cfg.get("kernel")
        avg_score = cfg.get("avg_score")
        if kernel is None or avg_score is None:
            continue
        if best_score is None:
            best_score = avg_score
            best_kernel = str(kernel).lower()
            best_cfg = cfg
            continue
        if maximize and avg_score > best_score:
            best_score = avg_score
            best_kernel = str(kernel).lower()
            best_cfg = cfg
        elif not maximize and avg_score < best_score:
            best_score = avg_score
            best_kernel = str(kernel).lower()
            best_cfg = cfg
    return best_kernel, best_cfg


def _ask_int(prompt, default=None):
    suffix = f" [defaut: {default}]: " if default is not None else ": "
    answer = input(prompt + suffix).strip()
    if not answer:
        return default
    try:
        return int(answer)
    except ValueError:
        return default


def _write_history_csv(out_dir, filename, history, meta=None):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    runtime = history.get("runtime") or list(range(1, len(history.get("best_fitness", [])) + 1))
    score_mean = history.get("score_mean", [])
    score_median = history.get("score_median", [])
    score_std = history.get("score_std", [])
    score_p2 = history.get("score_p2", [])
    score_p5 = history.get("score_p5", [])
    score_p10 = history.get("score_p10", [])
    score_p25 = history.get("score_p25", [])
    score_p50 = history.get("score_p50", [])
    score_p75 = history.get("score_p75", [])
    score_p90 = history.get("score_p90", [])
    score_p95 = history.get("score_p95", [])
    score_p98 = history.get("score_p98", [])
    rows = zip(
        runtime,
        history.get("best_fitness", []),
        history.get("avg_hamming", []),
        history.get("avg_l1", []),
        history.get("avg_entropy", []),
        score_mean,
        score_median,
        score_std,
        score_p2,
        score_p5,
        score_p10,
        score_p25,
        score_p50,
        score_p75,
        score_p90,
        score_p95,
        score_p98,
    )
    csv_path = os.path.join(out_dir, filename)
    with open(csv_path, "w") as f:
        if meta:
            for key, value in meta.items():
                f.write(f"# {key}: {value}\n")
        f.write(
            "step,best_fitness,avg_hamming,avg_l1,avg_entropy,"
            "mean,median,std,2%,5%,10%,25%,50%,75%,90%,95%,98%\n"
        )
        for (step, bf, ham, l1, ent, mean, median, std, p2, p5, p10, p25, p50, p75, p90, p95, p98) in rows:
            f.write(
                f"{step},{bf},{ham},{l1},{ent},"
                f"{mean},{median},{std},{p2},{p5},{p10},{p25},{p50},{p75},{p90},{p95},{p98}\n"
            )


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):

    # Support keeping the original variable names used previously; read them from Hydra cfg
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('running on device: ' + device)
    script_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    type_problem = cfg.problem.name if 'problem' in cfg and 'name' in cfg.problem else cfg.get('type_problem', 'QUBO')
    print(f"Running with problem type: {type_problem}")
    dim = (
        cfg.problem.n
        if 'problem' in cfg and 'n' in cfg.problem
        else cfg.problem.dim if 'problem' in cfg and 'dim' in cfg.problem else cfg.get('dim', 64)
    )
    type_instance = (
        cfg.problem.k
        if 'problem' in cfg and 'k' in cfg.problem
        else cfg.problem.type_instance if 'problem' in cfg and 'type_instance' in cfg.problem else cfg.get('type_instance', 1)
    )
    print(f"Running with dim={dim}, type_instance={type_instance}")
    nb_restarts = int(cfg.nb_restarts)
    nb_instances_test = int(cfg.nb_instances_test)
    seed = int(cfg.seed)
    def agent_val(key):
        try:
            return OmegaConf.select(cfg, f"agent.{key}")
        except Exception:
            return None

    lambda_ = int(agent_val("lambda") or cfg.get('lambda') or cfg.get('lambda_') or 10)
    verbose = bool(cfg.get('verbose', True))
    budget = int(cfg.get('budget', 10000))
    visualization_enabled = bool(cfg.get('visualization', True))
    advantage_cfg = agent_val("advantage") or cfg.get('advantage') or "baseline"
    no_interact = bool(agent_val("no_interact") or cfg.get("no_interact") or False)
    decay_enabled = bool(agent_val("decay") or cfg.get("decay") or False)
    if isinstance(advantage_cfg, DictConfig):
        advantage_cfg = OmegaConf.to_container(advantage_cfg, resolve=True)
    M = int(agent_val("M") or cfg.get('M') or 1)
    learning_rate = None
    typeStrategy = "PPO-EDA"
    kernel_name = str(agent_val("kernel") or cfg.get("kernel") or "hk").lower()
    ask_best_config = bool(cfg.get("ask_best_config", True))
    epsilon_override = None
    gamma_override = None
    bandwith_override = None
    best_cfg = None
    if decay_enabled:
        no_interact = False
        instance_name = f"{type_problem}_dim{dim}_t{type_instance}"
        summary_dir = os.path.join(repo_root, "results", "experiments", instance_name)
        best_kernel, best_cfg = _find_best_kernel_summary(summary_dir, type_problem)
        if best_cfg is None or best_kernel is None:
            print(f"[WARN] Aucun resume valide dans {summary_dir}. On garde la config actuelle.")
        else:
            kernel_name = best_kernel
            advantage_cfg = best_cfg.get("advantage", advantage_cfg)
            M = int(best_cfg.get("m", M))
            lambda_ = int(best_cfg.get("lambda", lambda_))
            epsilon_override = best_cfg.get("epsilon_svgd")
            gamma_override = best_cfg.get("gamma")
            bandwith_override = best_cfg.get("bandwith_kernel")
    elif ask_best_config and _ask_yes_no("Recuperer la meilleure config depuis results/experiments? (DEFAULT FALSE)", default=False):
        budget = _ask_int("Budget a utiliser", default=budget) or budget
        decay_enabled = _ask_yes_no(f"Mode decay? (actuel: {decay_enabled})", default=decay_enabled)
        if decay_enabled:
            no_interact = False
        else:
            no_interact = _ask_yes_no(f"Mode no_interact? (actuel: {no_interact})", default=no_interact)
        instance_name = f"{type_problem}_dim{dim}_t{type_instance}"
        summary_dir = os.path.join(repo_root, "results", "experiments", instance_name)
        if no_interact:
            summary_dir = os.path.join(summary_dir, "no_interact")
        best_kernel, best_cfg = _find_best_kernel_summary(summary_dir, type_problem)
        if best_cfg is None or best_kernel is None:
            print(f"[WARN] Aucun resume valide dans {summary_dir}. On garde la config actuelle.")
        else:
            kernel_name = best_kernel
            advantage_cfg = best_cfg.get("advantage", advantage_cfg)
            M = int(best_cfg.get("m", M))
            lambda_ = int(best_cfg.get("lambda", lambda_))
            epsilon_override = best_cfg.get("epsilon_svgd")
            gamma_override = best_cfg.get("gamma")
            bandwith_override = best_cfg.get("bandwith_kernel")
    write_budget_results = _ask_yes_no(
        "Enregistrer l'historique dans results/experiments? (DEFAULT FALSE)", default=False
    )
    kernel_cfg = _load_kernel_config(kernel_name, repo_root)
    if bandwith_override is not None:
        kernel_cfg["bandwith_kernel"] = bandwith_override
    kernel_lr = kernel_cfg.get("epsilon_svgd")
    kernel_gamma = kernel_cfg.get("gamma")
    kernel_bandwith_kernel = kernel_cfg.get("bandwith_kernel") or (kernel_cfg.get("params") or {}).get("bandwith_kernel")
    epsilon_svgd = float(
        agent_val("epsilon_svgd")
        or cfg.get('epsilon_svgd')
        or epsilon_override
        or kernel_lr
        or 0.5
    )
    learning_rate = epsilon_svgd
    svgd_gamma = float(
        agent_val("gamma")
        or cfg.get('gamma')
        or gamma_override
        or kernel_gamma
        or 10.0
    )
    decay_default_start_ratio = 0.0 if decay_enabled else 0.8
    decay_default_min_factor = 0.05 if decay_enabled else 0.1
    decay_start_ratio = float(
        agent_val("decay_start_ratio")
        or cfg.get("decay_start_ratio")
        or decay_default_start_ratio
    )
    decay_min_factor = float(
        agent_val("min_factor")
        or agent_val("decay_min_factor")
        or cfg.get("min_factor")
        or cfg.get("decay_min_factor")
        or decay_default_min_factor
    )
    if best_cfg:
        if best_cfg.get("decay_start_ratio") is not None:
            decay_start_ratio = float(best_cfg.get("decay_start_ratio"))
        if best_cfg.get("min_factor") is not None:
            decay_min_factor = float(best_cfg.get("min_factor"))
        elif best_cfg.get("decay_min_factor") is not None:
            decay_min_factor = float(best_cfg.get("decay_min_factor"))
    bandwith_kernel_suffix = ""
    if kernel_name in ("pk", "rbf"):
        bandwith_kernel_suffix = f", bandwith_kernel: {kernel_bandwith_kernel}"
    print(
        f"Using REINFORCE update. Number of agents: {M} with epsilon_svgd: {epsilon_svgd}, "
        f"λ: {lambda_}, svgd_gamma: {svgd_gamma}, advantage={advantage_cfg}, "
        f"kernel={kernel_name}{bandwith_kernel_suffix}, no_interact={no_interact}, bandwith_kernel: {kernel_bandwith_kernel}, decay_start_ratio: {decay_start_ratio}, decay_min_factor: {decay_min_factor}"
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    N = dim

    write_logs = bool(cfg.get('write_logs', False))
    pathResult = None
    block_size = None
    dummy_blocks = 0
    total_lambda = lambda_ * M if typeStrategy == "PPO-EDA" else lambda_

    if (type_problem == "QUBO"):

        # Instances live under source_code/instances in this repo; resolve absolute path
        # add trailing sep because downstream loader concatenates filenames
        instance_path = os.path.join(script_dir, "instances", "QUBO") + os.sep
        try:
            tensor_Q_test = getTensorInstances_QUBO(instance_path, nb_instances_test, nb_restarts, N, type_instance, device,
                                                    "test")
        except FileNotFoundError as e:
            # fallback to a default dimension if requested instances not available
            fallback_dim = 64
            print(f"Requested problem dim={N} not available; falling back to default dim={fallback_dim}.")
            N = fallback_dim
            dim = fallback_dim
            # recompute pathResult for fallback dim
            if write_logs:
                pathResult = os.path.join(repo_root, "results", "results_Multivariate-RL-EDA", typeStrategy, str(type_problem), str(dim), str(type_instance)) + os.sep
                os.makedirs(pathResult, exist_ok=True)
            tensor_Q_test = getTensorInstances_QUBO(instance_path, nb_instances_test, nb_restarts, N, type_instance, device,
                                                    "test")
    elif(type_problem == "NK"):

        D = 2
        vectorIndex = np.zeros((type_instance + 1))
        for i in range(type_instance + 1):
            vectorIndex[i] = D ** (type_instance - i)
        vectorIndex_th = torch.tensor(vectorIndex, dtype=torch.float32).to(device)

        nk_path = os.path.join(script_dir, "instances", "nk", str(dim), str(type_instance)) + os.sep
        tensor_matrix_locus, tensor_matrix_contrib, tensor_Q_test = getTensorInstances_NK(
            nk_path, nb_instances_test, nb_restarts, total_lambda, dim, D, type_instance, device
        )

    elif(type_problem == "NK3"):

        D = 3
        vectorIndex = np.zeros((type_instance + 1))
        for i in range(type_instance + 1):
            vectorIndex[i] = D ** (type_instance - i)
        vectorIndex_th = torch.tensor(vectorIndex, dtype=torch.float32).to(device)

        nk3_path = os.path.join(script_dir, "instances", "nk3", str(dim), str(type_instance)) + os.sep
        tensor_matrix_locus, tensor_matrix_contrib, tensor_Q_test = getTensorInstances_NK(
            nk3_path, nb_instances_test, nb_restarts, total_lambda, dim, D, type_instance, device
        )
    elif type_problem == "BLOCK":
        block_size = type_instance
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if N % block_size != 0:
            raise ValueError(f"dim={N} must be divisible by block_size={block_size}")
        dummy_blocks = int(cfg.problem.dummy_blocks) if "problem" in cfg and "dummy_blocks" in cfg.problem else 0



    factory = FactoryStrategyEA()


    if (type_problem == "NK3"):
        dim_variables = [3 for i in range(N)]
    else:
        dim_variables = None


    strategy = factory.createStrategyEA(
        typeStrategy,
        dim,
        lambda_,
        device,
        dim_variables,
        M,
        learning_rate=learning_rate,
        epsilon_svgd=epsilon_svgd,
        enable_visualization=visualization_enabled,
        svgd_gamma=svgd_gamma,
        decay_start_ratio=decay_start_ratio,
        decay_min_factor=decay_min_factor,
        decay_enabled=decay_enabled,
        advantage_cfg=advantage_cfg,
        kernel_config=kernel_cfg,
        no_interact=no_interact,
    ).to(device)
    if (type_problem == "QUBO"):
        result = get_Score_trajectoriesQUBO_cuda(
            strategy,
            N,
            nb_instances_test,
            nb_restarts,
            budget,
            lambda_,
            tensor_Q_test,
            device,
            verbose,
            enable_visualization=visualization_enabled,
            return_history=write_budget_results,
        )
        if write_budget_results:
            list_scores, history = result
        else:
            list_scores = result

    elif (type_problem == "NK" or type_problem == "NK3"):
        result = get_Score_trajectoriesNK_cuda(strategy, N,  type_instance, D, nb_instances_test, nb_restarts, 
                                               budget, lambda_,
                                               vectorIndex_th, tensor_matrix_locus,
                                               tensor_matrix_contrib, device, verbose,
                                               return_history=write_budget_results)
        if write_budget_results:
            list_scores, history = result
        else:
            list_scores = result
    elif type_problem == "BLOCK":
        result = get_Score_trajectoriesBLOCK_cuda(
            strategy,
            N,
            block_size,
            nb_instances_test,
            nb_restarts,
            budget,
            lambda_,
            device,
            verbose,
            enable_visualization=visualization_enabled,
            dummy_blocks=dummy_blocks,
            return_history=write_budget_results,
        )
        if write_budget_results:
            list_scores, history = result
        else:
            list_scores = result
        
    print(list_scores)
    average_test_score = np.mean(list_scores)

    print("average_test_score : " + str(average_test_score))
    if write_budget_results:
        instance_name = f"{type_problem}_dim{dim}_t{type_instance}"
        budget_dir = os.path.join(repo_root, "results", "experiments", instance_name, str(budget))
        if decay_enabled:
            filename = "decay.csv"
        else:
            filename = "no_interact.csv" if no_interact else "interact.csv"
        meta = dict(
            epsilon_svgd=epsilon_svgd,
            lambda_=lambda_,
            gamma=svgd_gamma,
            decay_start_ratio=decay_start_ratio,
            decay_min_factor=decay_min_factor,
            decay_enabled=decay_enabled,
            kernel=kernel_name,
            advantage=advantage_cfg,
            M=M,
        )
        _write_history_csv(budget_dir, filename, history, meta=meta)

if __name__ == '__main__':
    # Run hydra main
    main()
