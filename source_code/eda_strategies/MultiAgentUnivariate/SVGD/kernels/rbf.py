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
        # sigma :
        #   - si None : sigma sera estimé automatiquement (median heuristic)
        #   - sinon   : on utilise cette valeur fixe (float ou tensor)
        # facteur de largeur du noyau RBF (peut être fixé via la config)
        self.bandwith_kernel = bandwith_kernel

    def forward(self, Thetas):
        """
        X : (B, M, N)
        Y : (B, P, N)

        Retourne :
            K : (B, M, P) avec K[b, i, j] = exp( - bandwith_kernel * || X_{b,i} - Y_{b,j} ||^2 )
        """
        Thetas = Thetas.requires_grad_(True)

        B, M, N = Thetas.shape 
        # B : nombre d'instances 
        # M : nombre de particules
        # N : dimension des particules


        theta_i = Thetas.unsqueeze(2).repeat([1,1,M,1])  # (B, M, M, N)
        theta_j = Thetas.unsqueeze(1).repeat([1,M,1,1])  # (B, M, M, N)

        dnorm2 = ((theta_i - theta_j.detach()) ** 2).sum(dim=-1)

        if self.bandwith_kernel is None:
            # Estimation automatique de la bandwith via la median heuristic
            bandwith_kernel = adaptative_bandwith(dnorm2, eps=1e-8)
        else:
            bandwith_kernel = self.bandwith_kernel

        K = torch.exp(-bandwith_kernel * dnorm2)

        grad_Thetas = torch.zeros((B, M, N), device=Thetas.device)

        for i in range(M):
            Ki = K[:,:,i]
            vect_grad_Thetas, = torch.autograd.grad(Ki.sum(), Thetas, retain_graph=True)
            grad_Thetas[:,i,:] = torch.sum(vect_grad_Thetas, dim=1)
        return K, grad_Thetas
