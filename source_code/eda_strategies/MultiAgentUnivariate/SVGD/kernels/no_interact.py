import torch
import torch.nn as nn


class NoInteractKernel(nn.Module):
    """
    Kernel non-interactif : matrice identité et gradient nul.
    """

    def forward(self, Thetas, probs=None):
        """
        Thetas : (B, M, N)
        Retourne :
            K : (B, M, M) identité par batch
            grad_Thetas : (B, M, N) nul
        """
        B, M, N = Thetas.shape
        K = torch.eye(M, device=Thetas.device, dtype=Thetas.dtype).expand(B, M, M)
        grad_Thetas = torch.zeros((B, M, N), device=Thetas.device, dtype=Thetas.dtype)
        return K, grad_Thetas
