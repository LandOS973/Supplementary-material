import torch
import torch.nn as nn

from .utils import adaptative_bandwith


class RBF(nn.Module):
    """
    Kernel RBF pour tenseurs (B, M, N).

    Pour deux tenseurs X, Y de forme (B, M, N)
    ce module renvoie un tenseur K de forme (B, M, P) avec :

        K[b, i, j] = k(X[b, i, :], Y[b, j, :])
                   = exp( - bandwith_kernel * || X_{b,i} - Y_{b,j} ||^2 )

    """

    def __init__(self, bandwith_kernel):
        super().__init__()
        self.bandwith_kernel = bandwith_kernel

    def forward(self, Thetas, probs=None, support_thetas=None, support_probs=None):
        """
        X : (B, M, N)
        Y : (B, P, N)

        Retourne :
            K : (B, M, P) avec K[b, i, j] = exp( - bandwith_kernel * || X_{b,i} - Y_{b,j} ||^2 )
        """
        if support_thetas is not None:
            query = Thetas.requires_grad_(True)
            support = support_thetas.detach()
            if query.dim() + 1 != support.dim():
                raise ValueError(
                    f"support_thetas must have query dims + 1, got query={query.shape}, support={support.shape}."
                )

            if query.dim() == 4:
                query_expanded = query.unsqueeze(2)
                diff = query_expanded - support
                dnorm2 = (diff ** 2).sum(dim=-1).sum(dim=-1)
            else:
                query_expanded = query.unsqueeze(2)
                diff = query_expanded - support
                dnorm2 = (diff ** 2).sum(dim=-1)

            if self.bandwith_kernel is None:
                bandwith_kernel = adaptative_bandwith(dnorm2, eps=1e-8)
            else:
                bandwith_kernel = self.bandwith_kernel

            K = torch.exp(-bandwith_kernel * dnorm2)

            if query.dim() == 4:
                grad_Thetas = (
                    2.0
                    * bandwith_kernel
                    * diff
                    * K.unsqueeze(-1).unsqueeze(-1)
                ).sum(dim=2)
            else:
                grad_Thetas = (
                    2.0
                    * bandwith_kernel
                    * diff
                    * K.unsqueeze(-1)
                ).sum(dim=2)

            return K, grad_Thetas

        Thetas = Thetas.requires_grad_(True)

        if Thetas.dim() == 4:
            B, M, N, D = Thetas.shape
        else:
            B, M, N = Thetas.shape

        if Thetas.dim() == 4:
            theta_i = Thetas.unsqueeze(2).repeat([1, 1, M, 1,1])                
            theta_j = Thetas.unsqueeze(1).repeat([1, M, 1, 1,1])                

            dnorm2 = ((theta_i - theta_j.detach()) ** 2).sum(dim=-1).sum(dim=-1)


        else:
            theta_i = Thetas.unsqueeze(2).repeat([1,1,M,1])                
            theta_j = Thetas.unsqueeze(1).repeat([1,M,1,1])                

            dnorm2 = ((theta_i - theta_j.detach()) ** 2).sum(dim=-1)


        if self.bandwith_kernel is None:
            bandwith_kernel = adaptative_bandwith(dnorm2, eps=1e-8)
        else:
            bandwith_kernel = self.bandwith_kernel

        K = torch.exp(-bandwith_kernel * dnorm2)

        if Thetas.dim() == 4:
    
            grad_Thetas = torch.zeros((B, M, N,D), device=Thetas.device)


            for i in range(M):
                Ki = K[:,:,i]
                vect_grad_Thetas, = torch.autograd.grad(Ki.sum(), Thetas, retain_graph=True)
                grad_Thetas[:,i,:,:] = torch.sum(vect_grad_Thetas, dim=1)
                
        else:
            
            grad_Thetas = torch.zeros((B, M, N), device=Thetas.device)

            for i in range(M):
                Ki = K[:,:,i]
                vect_grad_Thetas, = torch.autograd.grad(Ki.sum(), Thetas, retain_graph=True)
                grad_Thetas[:,i,:] = torch.sum(vect_grad_Thetas, dim=1)
                
                
        return K, grad_Thetas
