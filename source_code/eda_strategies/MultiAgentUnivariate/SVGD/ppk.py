import torch
import torch.nn as nn


class PPK(nn.Module):
    """
    Kernel PPK pour tenseurs (B, M, N)

    Kppk(Theta_i, Theta_j) = prod_k=1 a N Theta_i[k] * Theta_j[k] + (1 - Theta_i[k]) * (1 - Theta_j[k])

    Kppk IGO(Theta_i, Theta_j) = prod_k=1 a N Theta_i[k] * (1 - Theta_j[k]) * Theta_j[k] + (1 - Theta_j[k]) * Kppk(Theta_i, Theta_j)
    """

    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps
        # attribute kept for compatibility with SVGD logic expecting kernels to expose last_gamma
        self.last_gamma = None

    def forward(self, X, Y):
        """
        X : (B, M, N)
        Y : (B, P, N)

        Retourne :
            K : (B, M, P) avec K[b, i, j] = 
        """
        X_exp = X.unsqueeze(2)  # (B, M, 1, N)
        Y_exp = Y.unsqueeze(1)  # (B, 1, P, N)

        term1 = torch.mul(X_exp, Y_exp)  # (B, M, P, N)
        term2 = torch.mul(1 - X_exp, 1 - Y_exp)  # (B, M, P, N)

        K = torch.prod(term1 + term2, dim=-1)  # (B, M, P)

        return K
