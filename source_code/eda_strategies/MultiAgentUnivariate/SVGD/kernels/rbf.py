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
        Standard:  Thetas (B, M, N)         → K (B, M, M),         grad (B, M, N)
        Standard:  Thetas (B, M, N, D)       → K (B, M, M),         grad (B, M, N, D)
        Support:   Thetas (B, L, N),         support_thetas (B, L, M, N)
                                              → K (B, L, M),         grad (B, L, M, N)
        Support:   Thetas (B, L, N, D),      support_thetas (B, L, M, N, D)
                                              → K (B, L, M),         grad (B, L, M, N, D)
        """
        if support_thetas is not None:
            return self._forward_support(Thetas, support_thetas)

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

    def _forward_support(self, query, support_thetas):
        """
        Cas partiel/async : query (B, L, N[, D]), support_thetas (B, L, M, N[, D]).

        Retourne :
            K         (B, L, M)         — k(support_j, query_i)
            grad_term (B, L, M, N[, D]) — ∂k/∂support_j (non sommé), pour le terme répulsion SVGD
        """
        categorical = query.dim() == 4

        # diff[b, i, j, ...] = query[b, i, ...] - support[b, i, j, ...]
        diff = query.unsqueeze(2) - support_thetas.detach()  # [B, L, M, N] ou [B, L, M, N, D]

        if categorical:
            dnorm2 = (diff ** 2).sum(dim=-1).sum(dim=-1)   # [B, L, M]
        else:
            dnorm2 = (diff ** 2).sum(dim=-1)               # [B, L, M]

        if self.bandwith_kernel is None:
            M_sup = support_thetas.size(2)
            vals = dnorm2.detach().flatten()
            median_val = torch.median(vals)
            denom = 2.0 * torch.log(
                torch.tensor(float(M_sup + 1), device=dnorm2.device, dtype=dnorm2.dtype)
            )
            h = median_val / denom
            sigma = torch.sqrt(h.clamp(min=1e-8))
            bandwith_kernel = 1.0 / (1e-8 + 2.0 * sigma ** 2)
        else:
            bandwith_kernel = self.bandwith_kernel

        K = torch.exp(-bandwith_kernel * dnorm2)  # [B, L, M]

        # ∂k(support_j, query_i)/∂support_j = 2h * (query_i - support_j) * k = 2h * diff * k
        if categorical:
            grad_term = 2.0 * bandwith_kernel * diff * K.unsqueeze(-1).unsqueeze(-1)
        else:
            grad_term = 2.0 * bandwith_kernel * diff * K.unsqueeze(-1)

        return K, grad_term
