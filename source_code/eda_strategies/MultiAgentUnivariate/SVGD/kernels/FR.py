import torch
import torch.nn as nn

from .utils import adaptative_bandwith


class FisherRaoKernel(nn.Module):
    """
    Kernel Fisher-Rao pour tenseurs (B, M, N).

    Pour deux tenseurs Thetas de forme (B, M, N), on calcule :
        d_FR(theta_i, theta_j) = sqrt( 4 * sum_k (arcsin(sqrt(p_i,k)) - arcsin(sqrt(p_j,k)))^2 )
    avec p = sigmoid(theta) (probs passées en paramètre).

    Le noyau est :
        K[b, i, j] = exp( - gamma * d_FR(theta_i, theta_j) )
                    * prod_k g(p_{i,k}) g(p_{j,k})
    où
        g(x) = exp( -1 / (tau^2 * x * (1 - x)) + 4 / tau^2 )
    """

    def __init__(self, bandwith_kernel=None, tau=10000.0):
        super().__init__()
        self.bandwith_kernel = bandwith_kernel
        self.tau = float(tau)

    def forward(self, Thetas, probs=None):
        """
        Thetas : (B, M, N)
        probs  : (B, M, N) (déjà calculées, typiquement sigmoid(Thetas))

        Retourne :
            K : (B, M, M)
            grad_Thetas : (B, M, N)
        """
        if probs is None:
            raise ValueError("Fisher-Rao kernel requires probs.")

        Thetas = Thetas.requires_grad_(True)

        B, M, N = Thetas.shape

        # d_FR = sqrt( 4 * sum_k (arcsin(sqrt(p_i,k)) - arcsin(sqrt(p_j,k)))^2 )
        angles = torch.asin(torch.sqrt(probs))  # arcsin(sqrt(p))
        angles_i = angles.unsqueeze(2)               # (B, M, 1, N)
        angles_j = angles.detach().unsqueeze(1)      # (B, 1, M, N)

        sq_sum = ((angles_i - angles_j) ** 2).sum(dim=-1)  # (B, M, M)
        d_fr = 2.0 * torch.sqrt(sq_sum)

        if self.bandwith_kernel is None:
            gamma = adaptative_bandwith(d_fr,  eps=1e-8)
        else:
            gamma = self.bandwith_kernel

        # g(x) = exp( -1 / (tau^2 * x * (1 - x)) + 4 / tau^2 )
        tau2 = self.tau ** 2
        g = torch.exp(-1.0 / (tau2 * probs * (1.0 - probs)) + (4.0 / tau2))  # (B, M, N)
        prod_g = g.prod(dim=-1)  # (B, M)
        g_pair = prod_g.unsqueeze(2) * prod_g.unsqueeze(1)  # (B, M, M)

        K = torch.exp(-gamma * d_fr) * g_pair

        grad_Thetas = torch.zeros((B, M, N), device=Thetas.device, dtype=Thetas.dtype)

        for i in range(M):
            Ki = K[:, :, i]
            vect_grad_Thetas, = torch.autograd.grad(Ki.sum(), Thetas, retain_graph=True)
            grad_Thetas[:, i, :] = torch.sum(vect_grad_Thetas, dim=1)
        return K, grad_Thetas
