import torch
import torch.nn as nn

class PPK(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Thetas, probs=None):
        if probs is None:
            raise ValueError("PPK kernel requires probs.")
        Thetas = Thetas.requires_grad_(True)

        if Thetas.dim() == 4:
            Theta_i = probs.unsqueeze(2)               
            Theta_j = probs.unsqueeze(1)               
            f = (Theta_i * Theta_j).sum(dim=-1)             
        else:
            Theta_i = probs.unsqueeze(2)             
            Theta_j = probs.unsqueeze(1)             
            f = Theta_i * Theta_j + (1 - Theta_i) * (1 - Theta_j)

        K = torch.prod(f, dim=-1)           

        grad_Thetas, = torch.autograd.grad(K.sum(), Thetas, create_graph=True)

        return K, grad_Thetas                         
