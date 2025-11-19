import math
import torch
import torch.nn as nn


class RBF(nn.Module):
    def __init__(self, sigma=None):
        super(RBF, self).__init__()
        self.sigma = sigma
        self.last_gamma = None

    def forward(self, X, Y):
        if X.dim() == 2:
            return self._forward_matrix(X, Y)
        elif X.dim() == 3:
            return self._forward_batch(X, Y)
        raise ValueError("RBF kernel expects tensors of shape (M, N) or (B, M, N)")

    def _compute_gamma(self, dnorm2, m):
        if self.sigma is None:
            median = torch.median(dnorm2.detach().flatten())
            h = median / (2 * math.log(m + 1))
            sigma_val = torch.sqrt(torch.clamp(h, min=1e-8))
        else:
            sigma_val = torch.tensor(self.sigma, device=dnorm2.device, dtype=dnorm2.dtype)
        gamma = 1.0 / (1e-8 + 2 * sigma_val ** 2)
        if not torch.is_tensor(gamma):
            gamma = torch.tensor(gamma, device=dnorm2.device, dtype=dnorm2.dtype)
        self.last_gamma = gamma
        return gamma

    def _forward_matrix(self, X, Y):
        XX = X @ X.t()
        XY = X @ Y.t()
        YY = Y @ Y.t()
        dnorm2 = -2 * XY + XX.diagonal().unsqueeze(1) + YY.diagonal().unsqueeze(0)
        gamma = self._compute_gamma(dnorm2, X.size(0))
        return torch.exp(-gamma * dnorm2)

    def _forward_batch(self, X, Y):
        XX = torch.matmul(X, X.transpose(-1, -2))
        XY = torch.matmul(X, Y.transpose(-1, -2))
        YY = torch.matmul(Y, Y.transpose(-1, -2))
        diag_x = XX.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
        diag_y = YY.diagonal(dim1=-2, dim2=-1).unsqueeze(-2)
        dnorm2 = -2 * XY + diag_x + diag_y
        gamma = self._compute_gamma(dnorm2, X.size(-2))
        return torch.exp(-gamma * dnorm2)
