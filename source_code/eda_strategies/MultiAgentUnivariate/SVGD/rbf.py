import torch.nn as nn
import numpy as np
import torch

class RBF(nn.Module):
  def __init__(self, sigma=None):
    super(RBF, self).__init__()

    self.sigma = sigma

  def forward(self, X, Y):
    XX = X.matmul(X.t())
    XY = X.matmul(Y.t())
    YY = Y.matmul(Y.t())

    dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

    if self.sigma is None:
        h = dnorm2.detach().view(-1).median()
        h = h / (2 * torch.log(torch.tensor(X.size(0) + 1.0, device=X.device)))
        sigma = h.clamp(min=1e-8).sqrt()
    else:
        sigma = self.sigma


    gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
    K_XY = (-gamma * dnorm2).exp()

    return K_XY