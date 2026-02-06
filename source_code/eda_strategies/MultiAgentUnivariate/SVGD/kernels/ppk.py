import torch
import torch.nn as nn

class PPK(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Thetas, probs=None):
        # X: (B,M,N), Y: (B,P,N)
        if probs is None:
            raise ValueError("PPK kernel requires probs.")
        Thetas = Thetas.requires_grad_(True)

        Theta_i = probs.unsqueeze(2)  # (B,M,1,N)
        Theta_j = probs.unsqueeze(1)  # (B,1,P,N)

        f = Theta_i * Theta_j + (1 - Theta_i) * (1 - Theta_j)

        K = torch.prod(f, dim=-1)  # (B,M,P)

        grad_Thetas, = torch.autograd.grad(K.sum(), Thetas, create_graph=True)

        return K, grad_Thetas   # grad_Thetas: (B,M,N)
