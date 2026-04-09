"""
Interactive timing script for SVGD-EDA (PPO-EDA) on CPU/GPU.
Measures wall-clock time for given instance counts (e.g. 1 and 100).
Generates both a console summary and a LaTeX table.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

SOURCE_CODE_DIR = Path(__file__).resolve().parents[1]
if str(SOURCE_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_CODE_DIR))

from eda_strategies.FactoryStrategyEA import FactoryStrategyEA
from environment.qubo import getTensorInstances_QUBO, get_Score_trajectoriesQUBO_cuda
from environment.nk import getTensorInstances_NK, get_Score_trajectoriesNK_cuda
from environment.blockwise import get_Score_trajectoriesBLOCK_cuda


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_CONFIG_NAME = "krbf__advglobalrankweighted__M7__L13__eps0p08__g0p015__ds0p03__dm0p01"

def _load_yaml(path: Path) -> dict:
    cfg = OmegaConf.load(str(path))
    return OmegaConf.to_container(cfg, resolve=True) or {}


def _load_kernel_config(kernel_name: str) -> dict:
    kernel_dir = CONFIG_DIR / "kernel"
    kernel_path = kernel_dir / f"{kernel_name}.yaml"
    if not kernel_path.exists():
        available = ", ".join(sorted(p.stem for p in kernel_dir.glob("*.yaml")))
        raise FileNotFoundError(
            f"Kernel config '{kernel_name}' introuvable dans {kernel_dir}. "
            f"Kernels disponibles: {available}"
        )
    cfg = _load_yaml(kernel_path)
    if "name" not in cfg:
        cfg["name"] = kernel_name
    return cfg


def _parse_float_token(raw: str) -> float:
    txt = raw.strip().lower()
    if "p" in txt and "." not in txt:
        txt = txt.replace("p", ".")
    return float(txt)


def _parse_config_string(raw: str, defaults: dict) -> dict:
    """
    Parse a compact config string like:
      krbf__advglobalrankweighted__M7__L13__eps0p08__g0p015__ds0p03__dm0p01
    """
    cfg = dict(defaults)
    if not raw:
        return cfg
    parts = [p for p in raw.strip().split("__") if p]
    for part in parts:
        low = part.lstrip("_").lower()
        if low.startswith("k"):
            cfg["kernel"] = low[1:]
        elif low.startswith("adv"):
            cfg["advantage"] = low[3:]
        elif low.startswith("m"):
            cfg["M"] = int(low[1:])
        elif low.startswith("l"):
            cfg["lambda"] = int(low[1:])
        elif low.startswith("eps"):
            cfg["epsilon_svgd"] = _parse_float_token(low[3:])
        elif low.startswith("g"):
            cfg["gamma"] = _parse_float_token(low[1:])
        elif low.startswith("ds"):
            cfg["decay_start_ratio"] = _parse_float_token(low[2:])
        elif low.startswith("dm"):
            cfg["decay_min_factor"] = _parse_float_token(low[2:])
        elif low in ("ni", "nointeract", "no_interact"):
            cfg["no_interact"] = True
        elif low in ("nr", "norepulsion", "no_repulsion"):
            cfg["no_repulsion"] = True
    return cfg


def _prepare_strategy(
    device: torch.device,
    dim: int,
    lambda_: int,
    m_agents: int,
    advantage_cfg,
    kernel_cfg: dict,
    no_interact: bool,
    no_repulsion: bool,
    decay_enabled: bool,
    decay_start_ratio: float,
    decay_min_factor: float,
    epsilon_svgd: float,
    svgd_gamma: float,
    dim_variables,
    is_nk3: bool,
):
    factory = FactoryStrategyEA()
    strategy = factory.createStrategyEA(
        "PPO-EDA",
        dim,
        lambda_,
        device,
        dim_variables,
        m_agents,
        learning_rate=epsilon_svgd,
        epsilon_svgd=epsilon_svgd,
        enable_visualization=False,
        svgd_gamma=svgd_gamma,
        decay_start_ratio=decay_start_ratio,
        decay_min_factor=decay_min_factor,
        decay_enabled=decay_enabled,
        advantage_cfg=advantage_cfg,
        kernel_config=kernel_cfg,
        no_interact=no_interact,
        no_repulsion=no_repulsion,
        is_nk3=is_nk3,
    ).to(device)
    return strategy


def _run_once(
    *,
    device: torch.device,
    type_problem: str,
    dim: int,
    type_instance: int,
    nb_instances: int,
    nb_restarts: int,
    budget: int,
    lambda_: int,
    m_agents: int,
    advantage_cfg,
    kernel_cfg: dict,
    no_interact: bool,
    no_repulsion: bool,
    decay_enabled: bool,
    decay_start_ratio: float,
    decay_min_factor: float,
    epsilon_svgd: float,
    svgd_gamma: float,
    dummy_blocks: int,
) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    total_lambda = lambda_ * m_agents
    dim_variables = [3 for _ in range(dim)] if type_problem == "NK3" else None

    strategy = _prepare_strategy(
        device=device,
        dim=dim,
        lambda_=lambda_,
        m_agents=m_agents,
        advantage_cfg=advantage_cfg,
        kernel_cfg=kernel_cfg,
        no_interact=no_interact,
        no_repulsion=no_repulsion,
        decay_enabled=decay_enabled,
        decay_start_ratio=decay_start_ratio,
        decay_min_factor=decay_min_factor,
        epsilon_svgd=epsilon_svgd,
        svgd_gamma=svgd_gamma,
        dim_variables=dim_variables,
        is_nk3=type_problem == "NK3",
    )

    if type_problem == "QUBO":
        instance_path = os.path.join(str(REPO_ROOT / "source_code" / "instances" / "QUBO")) + os.sep
        tensor_Q_test = getTensorInstances_QUBO(
            instance_path, nb_instances, nb_restarts, dim, type_instance, device, "test"
        )
        _ = get_Score_trajectoriesQUBO_cuda(
            strategy,
            dim,
            nb_instances,
            nb_restarts,
            budget,
            lambda_,
            tensor_Q_test,
            device,
            verbose=False,
            enable_visualization=False,
            return_history=False,
        )
    elif type_problem in ("NK", "NK3"):
        d = 3 if type_problem == "NK3" else 2
        vector_index = np.zeros((type_instance + 1))
        for i in range(type_instance + 1):
            vector_index[i] = d ** (type_instance - i)
        vector_index_th = torch.tensor(vector_index, dtype=torch.float32).to(device)
        nk_dir = "nk3" if type_problem == "NK3" else "nk"
        nk_path = os.path.join(str(REPO_ROOT / "source_code" / "instances" / nk_dir / str(dim) / str(type_instance))) + os.sep
        tensor_matrix_locus, tensor_matrix_contrib, tensor_Q_test = getTensorInstances_NK(
            nk_path, nb_instances, nb_restarts, total_lambda, dim, d, type_instance, device
        )
        _ = get_Score_trajectoriesNK_cuda(
            strategy,
            dim,
            type_instance,
            d,
            nb_instances,
            nb_restarts,
            budget,
            lambda_,
            vector_index_th,
            tensor_matrix_locus,
            tensor_matrix_contrib,
            device,
            verbose=False,
            enable_visualization=False,
            return_history=False,
        )
    elif type_problem == "BLOCK":
        block_size = type_instance
        _ = get_Score_trajectoriesBLOCK_cuda(
            strategy,
            dim,
            block_size,
            nb_instances,
            nb_restarts,
            budget,
            lambda_,
            device,
            verbose=False,
            enable_visualization=False,
            return_history=False,
            dummy_blocks=dummy_blocks,
        )
    else:
        raise ValueError(f"Probleme non supporte: {type_problem}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    return end - start


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def _write_latex_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols = [
        "problem",
        "dim",
        "type_instance",
        "device",
        "instances_req",
        "instances_used",
        "restarts",
        "budget",
        "M",
        "lambda",
        "kernel",
        "advantage",
        "epsilon_svgd",
        "gamma",
        "decay_start_ratio",
        "decay_min_factor",
        "no_interact",
        "no_repulsion",
        "time_sec_avg",
        "time_sec_min",
        "time_sec_max",
    ]
    header = " & ".join(_latex_escape(c) for c in cols) + " \\\\"
    lines = []
    lines.append("\\begin{tabular}{%s}" % ("l" * len(cols)))
    lines.append("\\hline")
    lines.append(header)
    lines.append("\\hline")
    for row in rows:
        values = []
        for c in cols:
            val = row.get(c, "")
            if isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(_latex_escape(str(val)))
        lines.append(" & ".join(values) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _discover_qubo_pairs(instance_dir: Path) -> list[tuple[int, int, int]]:
    pairs = {}
    if not instance_dir.exists():
        return []
    for fname in instance_dir.iterdir():
        if not fname.name.endswith(".json"):
            continue
        name = fname.name
        parts = name.split("_")
        if len(parts) < 7:
            continue
        try:
            n_val = int(parts[3])
            t_val = int(parts[5])
        except Exception:
            continue
        key = (n_val, t_val)
        pairs.setdefault(key, 0)
        pairs[key] += 1
    out = []
    for (n_val, t_val), count in sorted(pairs.items()):
        out.append((n_val, t_val, count))
    return out


def _discover_nk_pairs(base_dir: Path, d: int) -> list[tuple[int, int, int]]:
    pairs = []
    if not base_dir.exists():
        return pairs
    for dim_dir in sorted(base_dir.iterdir()):
        if not dim_dir.is_dir():
            continue
        try:
            dim_val = int(dim_dir.name)
        except Exception:
            continue
        for k_dir in sorted(dim_dir.iterdir()):
            if not k_dir.is_dir():
                continue
            try:
                k_val = int(k_dir.name)
            except Exception:
                continue
            pattern_prefix = f"nk_{dim_val}_{k_val}_{d}_"
            count = 0
            for f in k_dir.iterdir():
                if f.is_file() and f.name.startswith(pattern_prefix) and f.name.endswith(".txt"):
                    count += 1
            if count > 0:
                pairs.append((dim_val, k_val, count))
    return pairs


def main() -> None:
    print("Mesure du temps de calcul SVGD-EDA (PPO-EDA).")
    print("Le temps inclut le chargement des instances et l'execution complete du budget.")

    base_cfg = _load_yaml(CONFIG_DIR / "config.yaml")
    agent_cfg = _load_yaml(CONFIG_DIR / "agent" / "reinforce.yaml")

    defaults = {
        "kernel": str(agent_cfg.get("kernel", "hk")).lower(),
        "advantage": agent_cfg.get("advantage", "baseline"),
        "M": int(agent_cfg.get("M", 1)),
        "lambda": int(agent_cfg.get("lambda", 10)),
        "epsilon_svgd": agent_cfg.get("epsilon_svgd"),
        "gamma": agent_cfg.get("gamma"),
        "decay_start_ratio": float(agent_cfg.get("decay_start_ratio", 0.8)),
        "decay_min_factor": float(agent_cfg.get("min_factor", 0.1)),
        "no_interact": bool(agent_cfg.get("no_interact", False)),
        "no_repulsion": bool(agent_cfg.get("no_repulsion", False)),
        "decay": bool(agent_cfg.get("decay", False)),
    }

    config_str = (
        input(
            f"Config compacte (ex: {DEFAULT_CONFIG_NAME}) [default: {DEFAULT_CONFIG_NAME}]: "
        ).strip()
        or DEFAULT_CONFIG_NAME
    )
    cfg = _parse_config_string(config_str, defaults)

    kernel_name = str(cfg["kernel"]).lower()
    kernel_cfg = _load_kernel_config(kernel_name)
    epsilon_svgd = float(
        cfg.get("epsilon_svgd")
        or base_cfg.get("epsilon_svgd")
        or kernel_cfg.get("epsilon_svgd")
        or 0.5
    )
    svgd_gamma = float(
        cfg.get("gamma")
        or base_cfg.get("gamma")
        or kernel_cfg.get("gamma")
        or 10.0
    )
    cfg["epsilon_svgd"] = epsilon_svgd
    cfg["gamma"] = svgd_gamma

    print("=== Config deroulee ===")
    print(
        f"kernel={kernel_name} | advantage={cfg['advantage']} | M={cfg['M']} | "
        f"lambda={cfg['lambda']} | eps={epsilon_svgd} | gamma={svgd_gamma} | "
        f"decay_start={cfg['decay_start_ratio']} | decay_min={cfg['decay_min_factor']} | "
        f"no_interact={cfg['no_interact']} | no_repulsion={cfg['no_repulsion']}"
    )

    budget = int(base_cfg.get("budget", 50000))
    instance_counts = [1, 100]
    repeats = 3

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    else:
        print("[INFO] GPU non detecte. Mesure GPU ignoree.")

    problems = ["QUBO", "NK", "NK3", "BLOCK"]

    qubo_pairs = _discover_qubo_pairs(REPO_ROOT / "source_code" / "instances" / "QUBO")
    nk_pairs = _discover_nk_pairs(REPO_ROOT / "source_code" / "instances" / "nk", d=2)
    nk3_pairs = _discover_nk_pairs(REPO_ROOT / "source_code" / "instances" / "nk3", d=3)
    block_cfg = _load_yaml(CONFIG_DIR / "problem" / "blockwise.yaml")

    out_dir = REPO_ROOT / "results" / "timing"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "time_summary.csv"
    out_tex = out_dir / "time_summary.tex"

    rows = []
    for problem_name in problems:
        if problem_name == "QUBO":
            problem_pairs = [(n, t, cnt) for (n, t, cnt) in qubo_pairs]
            dummy_blocks = 0
        elif problem_name == "NK":
            problem_pairs = [(n, k, cnt) for (n, k, cnt) in nk_pairs]
            dummy_blocks = 0
        elif problem_name == "NK3":
            problem_pairs = [(n, k, cnt) for (n, k, cnt) in nk3_pairs]
            dummy_blocks = 0
        else:
            dim = int(block_cfg.get("n", 256))
            type_instance = int(block_cfg.get("k", 256))
            dummy_blocks = int(block_cfg.get("dummy_blocks", 0))
            problem_pairs = [(dim, type_instance, None)]

        for (dim, type_instance, available) in problem_pairs:
            for device in devices:
                for count in instance_counts:
                    target_count = int(count)
                    if target_count <= 0:
                        continue

                    if available is None:
                        max_unique_instances = min(target_count, 10)
                    else:
                        max_unique_instances = min(int(available), target_count, 10)

                    if max_unique_instances <= 0:
                        print(f"[SKIP] {problem_name} dim={dim} t={type_instance} (no instances)")
                        continue

                    if target_count == 1:
                        used_instances = 1
                        current_restarts = 1
                    else:
                        used_instances = max_unique_instances
                        current_restarts = max(1, int(round(target_count / float(used_instances))))

                    times = []
                    for _ in range(repeats):
                        try:
                            elapsed = _run_once(
                                device=device,
                                type_problem=problem_name,
                                dim=dim,
                                type_instance=type_instance,
                                nb_instances=used_instances,
                                nb_restarts=current_restarts,
                                budget=budget,
                                lambda_=int(cfg["lambda"]),
                                m_agents=int(cfg["M"]),
                                advantage_cfg=cfg["advantage"],
                                kernel_cfg=kernel_cfg,
                                no_interact=bool(cfg["no_interact"]),
                                no_repulsion=bool(cfg["no_repulsion"]),
                                decay_enabled=bool(cfg["decay"]),
                                decay_start_ratio=float(cfg["decay_start_ratio"]),
                                decay_min_factor=float(cfg["decay_min_factor"]),
                                epsilon_svgd=float(cfg["epsilon_svgd"]),
                                svgd_gamma=float(cfg["gamma"]),
                                dummy_blocks=dummy_blocks,
                            )
                        except (FileNotFoundError, OSError, ValueError) as exc:
                            print(f"[SKIP] {problem_name} dim={dim} t={type_instance}: {exc}")
                            elapsed = None
                        if elapsed is None:
                            times = []
                            break
                        times.append(elapsed)
                    if not times:
                        continue
                    avg_time = float(np.mean(times))
                    row = {
                        "device": str(device),
                        "problem": problem_name,
                        "dim": dim,
                        "type_instance": type_instance,
                        "instances_req": target_count,
                        "instances_used": used_instances,
                        "restarts": current_restarts,
                        "budget": budget,
                        "M": int(cfg["M"]),
                        "lambda": int(cfg["lambda"]),
                        "kernel": kernel_name,
                        "advantage": cfg["advantage"],
                        "epsilon_svgd": float(cfg["epsilon_svgd"]),
                        "gamma": float(cfg["gamma"]),
                        "no_interact": bool(cfg["no_interact"]),
                        "no_repulsion": bool(cfg["no_repulsion"]),
                        "decay": bool(cfg["decay"]),
                        "decay_start_ratio": float(cfg["decay_start_ratio"]),
                        "decay_min_factor": float(cfg["decay_min_factor"]),
                        "repeats": repeats,
                        "time_sec_avg": avg_time,
                        "time_sec_min": float(np.min(times)),
                        "time_sec_max": float(np.max(times)),
                    }
                    rows.append(row)
                    print(
                        "[ROW] "
                        f"problem={row['problem']} dim={row['dim']} t={row['type_instance']} "
                        f"device={row['device']} instances_req={row['instances_req']} instances_used={row['instances_used']} "
                        f"restarts={row['restarts']} budget={row['budget']} M={row['M']} lambda={row['lambda']} "
                        f"kernel={row['kernel']} advantage={row['advantage']} "
                        f"eps={row['epsilon_svgd']} gamma={row['gamma']} "
                        f"ds={row['decay_start_ratio']} dm={row['decay_min_factor']} "
                        f"no_interact={row['no_interact']} no_repulsion={row['no_repulsion']} "
                        f"time_avg={row['time_sec_avg']:.3f}s"
                    )

    if rows:
        write_header = not out_csv.exists()
        with out_csv.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        _write_latex_table(out_tex, rows)
        print(f"[INFO] Resume CSV: {out_csv}")
        print(f"[INFO] Tableau LaTeX: {out_tex}")


if __name__ == "__main__":
    main()
