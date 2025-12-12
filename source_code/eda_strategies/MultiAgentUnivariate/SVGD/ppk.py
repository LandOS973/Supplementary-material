import torch
import torch.nn as nn


class PPK(nn.Module):
    """
    Kernel PPK pour tenseurs (B, M, N)

    Kppk(Theta_i, Theta_j) = prod_k=1 a N Theta_i[k] * Theta_j[k] + (1 - Theta_i[k]) * (1 - Theta_j[k])

    Kppk IGO(Theta_i, Theta_j) = prod_k=1 a N Theta_i[k] * (1 - Theta_j[k]) * Theta_j[k] + (1 - Theta_j[k]) * Kppk(Theta_i, Theta_j)
    """

    def __init__(self):
        super().__init__()

    def forward(self, X, Y):
        # Le kernel est défini sur des probabilités dans [0, 1].
        # On travaille donc sur les logits passés au SVGD en les sigmoïdant (tempérés).
        X_prob = torch.sigmoid(X)
        Y_prob = torch.sigmoid(Y)

        X_exp = X_prob.unsqueeze(2)  # (B, M, 1, N)
        Y_exp = Y_prob.unsqueeze(1)  # (B, 1, P, N)

        term1 = torch.mul(X_exp, Y_exp)  # (B, M, P, N)
        term2 = torch.mul(1 - X_exp, 1 - Y_exp)  # (B, M, P, N)
        per_dim = term1 + term2  # (B, M, P, N)

        logK = torch.sum(per_dim, dim=-1) / per_dim.size(-1)  # (B, M, P)
        K = torch.exp(logK)  # (B, M, P)
  

        # ∂/∂X_d k(X,Y) = (2 Y_d - 1) * Π_{k≠d} f_k avec Y en probas.
        # Chaîne complète : x = sigmoid(logit) => d/dlogit = d/dx * x(1-x).
        base = (2.0 * Y_exp - 1.0) * K.unsqueeze(-1) / (per_dim + 1e-12)
        grad = base * X_exp * (1.0 - X_exp)
        return K, grad
