import torch
import torch.nn as nn


class ProbabilityKernel(nn.Module):
    """
    Kernel de probabilité pour tenseurs (B, M, N).
    Pour deux tenseurs X, Y de forme (B, M, N)
    ce module renvoie un tenseur K de forme (B, M, P) avec :
        K[b, i, j] = k(X[b, i, :], Y[b, j, :])
                   = exp( - gamma * || p(X_{b,i}) - p(Y_{b,j}) ||^2 )
    où p(X) = sigmoid(X) est le vecteur des probabilités associées à l'agent X.
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


        dnorm2 = ((probs_i - probs_j.detach()) ** 2).sum(dim=-1) # (B, M, M)

        gamma = 1


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
