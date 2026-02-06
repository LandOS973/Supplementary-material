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
        # largeur du noyau appliqué sur les probabilités
        self.bandwith_kernel = bandwith_kernel

    def forward(self, Thetas, probs=None):
        """
        Thetas : (B, M, N)
        """
        if probs is None:
            raise ValueError("Probability kernel requires probs.")
        Thetas = Thetas.requires_grad_(True)

        B, M, N = Thetas.shape

        probs_i = probs.unsqueeze(2)  # (B, M, 1, N)
        probs_j = probs.unsqueeze(1)  # (B, 1, M, N)

        dnorm2 = ((probs_i - probs_j.detach()) ** 2).sum(dim=-1) # (B, M, M)

        if self.bandwith_kernel is None:
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





        ########

        # Thetas = Thetas.requires_grad_(True)
        #
        #
        #
        # probs_i = torch.sigmoid(Thetas).unsqueeze(2)  # (B, M, 1, N)
        # probs_j = torch.sigmoid(Thetas).unsqueeze(1)  # (B, 1, M, N)
        #
        # # hamming = probs_i + probs_j - 2 * probs_i * probs_j  # (B, M, M, N)
        # #
        # # D = hamming.sum(dim=-1)  # (B, M, M)
        # # N = Thetas.size(-1)
        # # K =((N - D) / (N))
        # #
        # # print("K v2")
        # # print(K[0][:5,:5])
        #
        # grad_Thetas = torch.zeros((B, M, N), device=Thetas.device)
        #
        # for i in range(M):
        #
        #     sum = 0
        #
        #     for j in range(M):
        #
        #         hamming = probs_i[:,i,:,:].detach() + probs_j[:,:,j,:] - 2 * probs_i[:,i,:,:].detach() * probs_j[:,:,j,:]
        #         D = hamming.sum(dim=-1)
        #         Kij = ((N - D) / (N))
        #
        #         vect_grad_Thetas_j, = torch.autograd.grad(Kij.sum(), Thetas, retain_graph=True)
        #         Thetas.grad = None
        #         sum += vect_grad_Thetas_j[:,j,:]
        #
        #
        #     grad_Thetas[:,i,:] = sum
        #
        #
        # print("grad_Thetas version 2")
        # print(grad_Thetas[0][:5,:5])
