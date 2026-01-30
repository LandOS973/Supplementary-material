import torch


def adaptative_bandwith(dist, eps=1e-8):
    """
    Calcule un facteur de largeur via la median heuristic.
    dist: (B, M, M) matrice de distances.
    """
    B, M, _ = dist.shape
    # On extrait les valeurs hors diagonale
    mask = ~torch.eye(M, device=dist.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    vals = dist.detach()[mask]

    median = torch.median(vals)
    denom = 2.0 * torch.log(torch.tensor(float(M + 1), device=dist.device, dtype=dist.dtype))
    h = median / denom
    sigma = torch.sqrt(h)
    gamma = 1.0 / (eps + 2.0 * sigma ** 2)
    return gamma
