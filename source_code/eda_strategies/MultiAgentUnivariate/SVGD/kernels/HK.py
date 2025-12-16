import torch
import torch.nn as nn


class HammingKernel(nn.Module):
    """
    Kernel basé sur la similarité de Hamming.

    Pour deux ensembles d'agents X et Y de forme (B, M, N) et (B, P, N),
    on calcule la distance moyenne de Hamming attendue entre chaque paire
    d'agents puis le kernel k(i, j) = N - D_{i, j}.

    Avec p_i = sigmoid(theta_i) les probabilités Bernoulli de l'agent i :
        D_{i, j} = Σ_k (p_{i,k} + p_{j,k} - 2 p_{i,k} p_{j,k})
    """

    def __init__(self):
        super().__init__()

    def forward(self, Thetas):
        """
        Thetas : (B, M, N)
        """
        Thetas = Thetas.requires_grad_(True)

        probs_i = torch.sigmoid(Thetas).unsqueeze(2)  # (B, M, 1, N)
        probs_j = torch.sigmoid(Thetas).unsqueeze(1)  # (B, 1, P, N)

        hamming = probs_i + probs_j - 2 * probs_i * probs_j  # (B, M, P, N)
        D = hamming.sum(dim=-1)  # (B, M, P)

        N = Thetas.size(-1) 
        K =((N - D) / N)# (B, M, P)

        grad_Thetas, = torch.autograd.grad(K.sum(), Thetas, create_graph=True)
        grad_term = -grad_Thetas
        return K, grad_term