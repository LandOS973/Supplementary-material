import torch
import torch.nn as nn

class PPK(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, Y):
        # X: (B,M,N), Y: (B,P,N)
        X = X.requires_grad_(True)

        Theta_i = torch.sigmoid(X).unsqueeze(2)  # (B,M,1,N)
        Theta_j = torch.sigmoid(Y).unsqueeze(1)  # (B,1,P,N)

        f = Theta_i * Theta_j + (1 - Theta_i) * (1 - Theta_j)

        K = torch.prod(f, dim=-1)  # (B,M,P)

        grad_X, = torch.autograd.grad(K.sum(), X, create_graph=True)

        return K, grad_X   # grad_X: (B,M,N)
