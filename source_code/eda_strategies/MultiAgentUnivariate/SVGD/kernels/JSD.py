import torch
import torch.nn as nn

from .utils import adaptative_bandwith

class JSD(nn.Module):
    def __init__(self, bandwith_kernel=None):
        super().__init__()
        self.bandwith_kernel = bandwith_kernel

    def forward(self, Thetas):
        Thetas = Thetas.requires_grad_(True)
        B, M, N = Thetas.shape

        # p_i avec gradient
        probs_i = torch.sigmoid(Thetas).unsqueeze(2)             # (B, M, 1, N)
        # p_j sans gradient (detach) comme dans ton code
        probs_j = torch.sigmoid(Thetas.detach()).unsqueeze(1)    # (B, 1, M, N)

        # clamp pour éviter log(0)
        pi = torch.clamp(probs_i, 1e-6, 1.0 - 1e-6)
        pj = torch.clamp(probs_j, 1e-6, 1.0 - 1e-6)
        m = 0.5 * (pi + pj)
        m = torch.clamp(m, 1e-6, 1.0 - 1e-6)

        kl_pm = pi * torch.log(pi / m) + (1.0 - pi) * torch.log((1.0 - pi) / (1.0 - m))
        kl_qm = pj * torch.log(pj / m) + (1.0 - pj) * torch.log((1.0 - pj) / (1.0 - m))

        jsd = 0.5 * (kl_pm + kl_qm)                              # (B, M, M, N)
        dist = jsd.sum(dim=-1) / float(N)                        # (B, M, M) normalisé

        # ===== median heuristic =====
        if self.bandwith_kernel is None:
            gamma = adaptative_bandwith(dist, eps=1e-6)
        else:
            gamma = self.bandwith_kernel

        # Kernel
        K = torch.exp(-gamma * dist)                             # (B, M, M)

        # ===== Gradient (même style que ton code) =====
        grad_Thetas = torch.zeros((B, M, N), device=Thetas.device, dtype=Thetas.dtype)

        for i in range(M):
            Ki = K[:, :, i]  # même convention que toi
            vect_grad_Thetas, = torch.autograd.grad(Ki.sum(), Thetas, retain_graph=True)
            grad_Thetas[:, i, :] = torch.sum(vect_grad_Thetas, dim=1)
        return K, grad_Thetas
