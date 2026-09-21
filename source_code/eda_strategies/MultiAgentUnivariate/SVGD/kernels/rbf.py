import torch
import torch.nn as nn

from .utils import adaptative_bandwith


class RBF(nn.Module):
    """
    Kernel RBF pour tenseurs (B, M, N).

    Pour deux tenseurs X, Y de forme (B, M, N)
    ce module renvoie un tenseur K de forme (B, M, P) avec :

        K[b, i, j] = k(X[b, i, :], Y[b, j, :])
                   = exp( - bandwith_kernel * || X_{b,i} - Y_{b,j} ||^2 )

    """

    def __init__(self):
        super().__init__()

    def forward(self, Thetas, probs=None):
        """
        Thetas (B, M, N)    → K (B, M, M), grad (B, M, N)
        Thetas (B, M, N, D) → K (B, M, M), grad (B, M, N, D)
        """
        categorical = Thetas.dim() == 4

        # diff[b, i, j, ...] = theta_i - theta_j
        diff = Thetas.unsqueeze(2) - Thetas.unsqueeze(1)

        if categorical:
            dnorm2 = (diff ** 2).sum(dim=-1).sum(dim=-1)
        else:
            dnorm2 = (diff ** 2).sum(dim=-1)

        bandwith_kernel = adaptative_bandwith(dnorm2, eps=1e-8)

        K = torch.exp(-bandwith_kernel * dnorm2)

        # Forme fermee du gradient RBF (equivalente a la boucle autograd) :
        # grad_Thetas[i] = 2*gamma * sum_j K[i,j] * (theta_i - theta_j)
        if categorical:
            weight = bandwith_kernel.unsqueeze(-1)
            grad_Thetas = 2.0 * weight * (K.unsqueeze(-1).unsqueeze(-1) * diff).sum(dim=2)
        else:
            grad_Thetas = 2.0 * bandwith_kernel * (K.unsqueeze(-1) * diff).sum(dim=2)

        return K, grad_Thetas