import torch
import torch.nn as nn


class PPK(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, Y):
        Thetai = torch.sigmoid(X)
        Thetaj  = torch.sigmoid(Y)

        Ti = Thetai.unsqueeze(2)
        Tj = Thetaj.unsqueeze(1)

        f = Ti * Tj + (1 - Ti) * (1 - Tj)

        # log-kernel au lieu de prod (plus stable numériquement)
        logK = torch.log(f).sum(dim=-1)
        # Centrage pour stabilité numérique
        logK = logK - logK.mean(dim=2, keepdim=True)
        # gradient de logK
        grad_theta = (2.0 * Tj - 1.0) / f
        grad_X = grad_theta * Thetai.unsqueeze(2) * (1 - Thetai.unsqueeze(2))
        return logK, grad_X
