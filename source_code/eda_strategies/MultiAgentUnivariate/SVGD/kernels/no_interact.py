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
        if Thetas.dim() == 4:
            B, M, _, _ = Thetas.shape
        else:
            B, M, _ = Thetas.shape
        grad_Thetas = torch.zeros_like(Thetas)
            
        K = torch.eye(M, device=Thetas.device, dtype=Thetas.dtype).expand(B, M, M)
        
        
        return K, grad_Thetas
