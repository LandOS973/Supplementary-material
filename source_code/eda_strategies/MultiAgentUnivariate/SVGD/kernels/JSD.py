import torch
import torch.nn as nn

class JSD(nn.Module):
    def __init__(self):
        super().__init__()

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
        mask = ~torch.eye(M, device=Thetas.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)      
        vals = dist.detach()[mask] 
        mediane = torch.median(vals)
        denom = 2.0 * torch.log(torch.tensor(float(M + 1), device=Thetas.device, dtype=Thetas.dtype))
        h = (mediane / denom)
        sigma = torch.sqrt(h)
        gamma = 1.0 / (1e-6 + 2.0 * sigma ** 2)

        # Kernel
        K = torch.exp(-gamma * dist)                             # (B, M, M)

        # ===== Gradient (même style que ton code) =====
        grad_Thetas = torch.zeros((B, M, N), device=Thetas.device, dtype=Thetas.dtype)

        for i in range(M):
            Ki = K[:, :, i]  # même convention que toi
            vect_grad_Thetas, = torch.autograd.grad(Ki.sum(), Thetas, retain_graph=True)
            grad_Thetas[:, i, :] = torch.sum(vect_grad_Thetas, dim=1)

        print(K[0])

        return K, grad_Thetas
