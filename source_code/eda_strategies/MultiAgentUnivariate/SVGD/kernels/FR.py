import torch
import torch.nn as nn

from .utils import adaptative_bandwith


class FisherRaoKernel(nn.Module):
    """
    Kernel Fisher-Rao pour tenseurs (B, M, N).

    Pour deux tenseurs Thetas de forme (B, M, N), on calcule :
        d_FR(theta_i, theta_j) = sqrt( 4 * sum_k (arcsin(sqrt(p_i,k)) - arcsin(sqrt(p_j,k)))^2 )
    avec p = sigmoid(theta) (probs passées en paramètre).

    Le noyau est :
        K[b, i, j] = exp( - gamma * d_FR(theta_i, theta_j) )
                    * prod_k g(p_{i,k}) g(p_{j,k})
    où
        g(x) = exp( -1 / (tau^2 * x * (1 - x)) + 4 / tau^2 )
    """

    def __init__(self, bandwith_kernel=None, tau=10000.0):
        super().__init__()
        self.bandwith_kernel = bandwith_kernel
        self.tau = float(tau)

    def forward(self, Thetas, probs=None):
        """
        Thetas : (B, M, N)
        probs  : (B, M, N) (déjà calculées, typiquement sigmoid(Thetas))

        Retourne :
            K : (B, M, M)
            grad_Thetas : (B, M, N)
        """
        if probs is None:
            raise ValueError("Fisher-Rao kernel requires probs.")

        Thetas = Thetas.requires_grad_(True)

        if Thetas.dim() == 4:
            B, M, N, D = Thetas.shape
        else:
            B, M, N = Thetas.shape

        if Thetas.dim() == 4:
            eps = 1e-7
            sqrt_probs = torch.sqrt(probs)
            sqrt_i = sqrt_probs.unsqueeze(2)                   
            sqrt_j = sqrt_probs.detach().unsqueeze(1)                   
            dot = (sqrt_i * sqrt_j).sum(dim=-1).clamp(min=eps, max=1.0 - eps)                
            angles = torch.acos(dot)                
            sq_sum = (angles ** 2).sum(dim=-1)             
            d_fr = 2.0 * torch.sqrt(sq_sum)
        else:
            angles = torch.asin(torch.sqrt(probs))                   
            angles_i = angles.unsqueeze(2)                             
            angles_j = angles.detach().unsqueeze(1)                    

            sq_sum = ((angles_i - angles_j) ** 2).sum(dim=-1)             
            d_fr = 2.0 * torch.sqrt(sq_sum)

        if self.bandwith_kernel is None:
            gamma = adaptative_bandwith(d_fr,  eps=1e-8)
        else:
            gamma = self.bandwith_kernel

        if Thetas.dim() == 4:
            tau2 = self.tau ** 2
            g = torch.exp(-1.0 / (tau2 * probs * (1.0 - probs)) + (4.0 / tau2))                
            mask = getattr(self, "mask", None)
            if mask is not None:
                mask = mask.to(probs.device, probs.dtype)
                g = g * mask + (1.0 - mask)
            g_var = g.prod(dim=-1)             
            prod_g = g_var.prod(dim=-1)          
            g_pair = prod_g.unsqueeze(2) * prod_g.unsqueeze(1)             

            K = torch.exp(-gamma * d_fr) * g_pair
            grad_Thetas = torch.zeros_like(Thetas)
            for i in range(M):
                Ki = K[:, :, i]
                vect_grad_Thetas, = torch.autograd.grad(Ki.sum(), Thetas, retain_graph=True)
                grad_Thetas[:, i, :, :] = torch.sum(vect_grad_Thetas, dim=1)
        else:
            tau2 = self.tau ** 2
            g = torch.exp(-1.0 / (tau2 * probs * (1.0 - probs)) + (4.0 / tau2))             
            prod_g = g.prod(dim=-1)          
            g_pair = prod_g.unsqueeze(2) * prod_g.unsqueeze(1)             

            K = torch.exp(-gamma * d_fr) * g_pair

            grad_Thetas = torch.zeros((B, M, N), device=Thetas.device, dtype=Thetas.dtype)

            for i in range(M):
                Ki = K[:, :, i]
                vect_grad_Thetas, = torch.autograd.grad(Ki.sum(), Thetas, retain_graph=True)
                grad_Thetas[:, i, :] = torch.sum(vect_grad_Thetas, dim=1)
        return K, grad_Thetas
