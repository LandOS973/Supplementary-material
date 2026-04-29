import torch


def adaptative_bandwith(dist, eps=1e-3):
    """
    Calcule un facteur de largeur via la median heuristic.
    dist: (B, M, P) matrice de distances, carrée ou rectangulaire.
    """
    B, M, P = dist.shape
    detached = dist.detach()

    if M == P:
        mask = ~torch.eye(M, device=dist.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
        vals = detached[mask]
    else:
        vals = detached.reshape(-1)

    if vals.numel() == 0:
        return torch.as_tensor(1.0 / eps, device=dist.device, dtype=dist.dtype)

    median = torch.median(vals)
    reference_size = max(M, P)
    denom = 2.0 * torch.log(torch.tensor(float(reference_size + 1), device=dist.device, dtype=dist.dtype))
    h = median / denom
    sigma = torch.sqrt(h)
    gamma = 1.0 / (eps + 2.0 * sigma ** 2)
    return gamma
