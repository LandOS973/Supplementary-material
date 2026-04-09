import torch
import torch.nn as nn
from .utils import adaptative_bandwith


class HammingKernel(nn.Module):
    """
    Kernel basé sur la similarité de Hamming.

    Pour deux ensembles d'agents X et Y de forme (B, M, N) et (B, P, N),
    on calcule la distance moyenne de Hamming attendue entre chaque paire
    d'agents puis le kernel k(i, j) = N - D_{i, j}.

    Avec p_i = sigmoid(theta_i) les probabilités Bernoulli de l'agent i :
        D_{i, j} = Σ_k (p_{i,k} + p_{j,k} - 2 p_{i,k} p_{j,k})
    """

    def __init__(self, bandwith_kernel=None):
        super().__init__()
        self.bandwith_kernel = bandwith_kernel

    def forward(self, Thetas, probs=None):
        """
        Thetas : (B, M, N)
        """
        if probs is None:
            raise ValueError("Hamming kernel requires probs.")
        Thetas = Thetas.requires_grad_(True)

        if Thetas.dim() == 4:
            B, M, N, D = Thetas.shape
            probs_i = probs.unsqueeze(2)                   
            probs_j = probs.detach().unsqueeze(1)                   
            match = (probs_i * probs_j).sum(dim=-1)                
            hamming = 1.0 - match                
            Dm = hamming.sum(dim=-1)             
            dist = (N - Dm) / float(N)             
        else:
            B, M, N = Thetas.shape
            probs_i = probs.unsqueeze(2)                
            probs_j = probs.detach().unsqueeze(1)                
            hamming = probs_i + probs_j.detach() - 2 * probs_i * probs_j.detach()                
            Dm = hamming.sum(dim=-1)             
            dist = (N - Dm) / float(N)             

        if self.bandwith_kernel is None:
            bandwith_kernel = adaptative_bandwith(dist, eps=1e-8)
        else:
            bandwith_kernel = self.bandwith_kernel

        K = torch.exp(-bandwith_kernel * dist)

        grad_Thetas = torch.zeros_like(Thetas)
        for i in range(M):
            Ki = K[:, :, i]
            vect_grad_Thetas, = torch.autograd.grad(Ki.sum(), Thetas, retain_graph=True)
            if Thetas.dim() == 4:
                grad_Thetas[:, i, :, :] = torch.sum(vect_grad_Thetas, dim=1)
            else:
                grad_Thetas[:, i, :] = torch.sum(vect_grad_Thetas, dim=1)

        return K, grad_Thetas






