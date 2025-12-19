import torch
import torch.nn as nn


class ProbabilityKernel(nn.Module):
    """
    Kernel basé sur la similarité de Hamming.

    Pour deux ensembles d'agents X et Y de forme (B, M, N) et (B, P, N),
    on calcule la distance moyenne de Hamming attendue entre chaque paire
    d'agents puis le kernel k(i, j) = N - D_{i, j}.

    Avec p_i = sigmoid(theta_i) les probabilités Bernoulli de l'agent i :
        D_{i, j} = Σ_k (p_{i,k} + p_{j,k} - 2 p_{i,k} p_{j,k})
    """

    def __init__(self):
        super().__init__()

    def forward(self, Thetas):
        """
        Thetas : (B, M, N)
        """
        Thetas = Thetas.requires_grad_(True)

        B, M, N = Thetas.shape


        theta_i = Thetas.unsqueeze(2).repeat([1,1,M,1])  # (B, M, M, N)
        theta_j = Thetas.unsqueeze(1).repeat([1,M,1,1])  # (B, M, M, N)

        probs_i = torch.sigmoid(theta_i)
        probs_j = torch.sigmoid(theta_j)


        dnorm2 = ((probs_i - probs_j.detach()) ** 2).sum(dim=-1)

        gamma = 0.01


        K = torch.exp(-gamma * dnorm2)

        vect_grad, = torch.autograd.grad(K.sum(), theta_i, retain_graph=True)  # (B, M, M, N)

        grad_Thetas = vect_grad.sum(dim=1)  # (B, M, N)


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
