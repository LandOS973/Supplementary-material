from __future__ import annotations

import urllib.request
from pathlib import Path

import torch


def load_gset_matrix(
    instance_name: str,
    url_base: str,
    device: str,
    base_dir: str | Path | None = None,
) -> tuple[torch.Tensor, int, int]:
    if base_dir is None:
        script_dir = Path(__file__).resolve().parent.parent
    else:
        script_dir = Path(base_dir).resolve()
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
    if agents_batch.dim() != 2:
        raise ValueError(f"agents_batch must be 2D (B, N), got shape={tuple(agents_batch.shape)}")
    x = agents_batch.to(dtype=adjacency_matrix.dtype)
    x_bar = 1.0 - x
    ax = torch.matmul(x, adjacency_matrix)
    return torch.sum(ax * x_bar, dim=1)
