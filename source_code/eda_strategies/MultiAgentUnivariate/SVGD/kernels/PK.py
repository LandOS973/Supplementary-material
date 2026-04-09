import torch
import torch.nn as nn

from .utils import adaptative_bandwith


class ProbabilityKernel(nn.Module):
    """
    Kernel de probabilité pour tenseurs (B, M, N).
    Pour deux tenseurs X, Y de forme (B, M, N)
    ce module renvoie un tenseur K de forme (B, M, P) avec :
        K[b, i, j] = k(X[b, i, :], Y[b, j, :])
                   = exp( - bandwith_kernel * || p(X_{b,i}) - p(Y_{b,j}) ||^2 )
    où p(X) = sigmoid(X) est le vecteur des probabilités associées à l'agent X.
    """

    def __init__(self, bandwith_kernel=1.0):
        super().__init__()
        self.bandwith_kernel = bandwith_kernel

    def forward(self, Thetas, probs=None):
        """
        Thetas : (B, M, N)
        """
        if probs is None:
            raise ValueError("Probability kernel requires probs.")
        Thetas = Thetas.requires_grad_(True)

        if Thetas.dim() == 4:
            B, M, N, D = Thetas.shape
            probs_i = probs.unsqueeze(2)                   
            probs_j = probs.unsqueeze(1)                   
            dnorm2 = ((probs_i - probs_j.detach()) ** 2).sum(dim=-1).sum(dim=-1)             
        else:
            B, M, N = Thetas.shape
            probs_i = probs.unsqueeze(2)                
            probs_j = probs.unsqueeze(1)                
            dnorm2 = ((probs_i - probs_j.detach()) ** 2).sum(dim=-1)             

        if self.bandwith_kernel is None:
            bandwith_kernel = adaptative_bandwith(dnorm2, eps=1e-8)
        else:
            bandwith_kernel = self.bandwith_kernel

        K = torch.exp(-bandwith_kernel * dnorm2)

        grad_Thetas = torch.zeros_like(Thetas)

        for i in range(M):
            Ki = K[:,:,i]
            vect_grad_Thetas, = torch.autograd.grad(Ki.sum(), Thetas, retain_graph=True)
            if Thetas.dim() == 4:
                grad_Thetas[:, i, :, :] = torch.sum(vect_grad_Thetas, dim=1)
            else:
                grad_Thetas[:, i, :] = torch.sum(vect_grad_Thetas, dim=1)
        return K, grad_Thetas






