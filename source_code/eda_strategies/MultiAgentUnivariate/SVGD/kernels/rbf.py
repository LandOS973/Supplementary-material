import math
import torch
import torch.nn as nn


class RBF(nn.Module):
    """
    Kernel RBF pour tenseurs (B, M, N).

    Pour deux tenseurs X, Y de forme (B, M, N)
    ce module renvoie un tenseur K de forme (B, M, P) avec :

        K[b, i, j] = k(X[b, i, :], Y[b, j, :])
                   = exp( - gamma * || X_{b,i} - Y_{b,j} ||^2 )

    où gamma = 1 / (2 * sigma^2), avec sigma soit fixé, soit estimé par
    "median heuristic" à partir des distances dans dnorm2.
    """

    def __init__(self, sigma=None):
        super().__init__()
        # sigma :
        #   - si None : sigma sera estimé automatiquement (median heuristic)
        #   - sinon   : on utilise cette valeur fixe (float ou tensor)
        self.sigma = sigma

    def forward(self, Thetas):
        """
        X : (B, M, N)
        Y : (B, P, N)

        Retourne :
            K : (B, M, P) avec K[b, i, j] = exp( - gamma * || X_{b,i} - Y_{b,j} ||^2 )
        """
        Thetas = Thetas.requires_grad_(True)

        B, M, N = Thetas.shape



        theta_i = Thetas.unsqueeze(2).repeat([1,1,M,1])  # (B, M, M, N)
        theta_j = Thetas.unsqueeze(1).repeat([1,M,1,1])  # (B, M, M, N)

        dnorm2 = ((theta_i - theta_j.detach()) ** 2).sum(dim=-1)

        gamma = 0.01


        K = torch.exp(-gamma * dnorm2)

        vect_grad, = torch.autograd.grad(K.sum(), theta_i, retain_graph=True)  # (B, M, M, N)

        grad_Thetas = vect_grad.sum(dim=1)  # (B, M, N)


        return K, grad_Thetas



        # # print(K[0][:5][:5])
        #
        # grad_Thetas = torch.zeros((B, M, N), device=Thetas.device)
        #
        # for i in range(M):
        #     Ki = K[:,:,i]
        #     vect_grad_Thetas, = torch.autograd.grad(Ki.sum(), Thetas, retain_graph=True)
        #     grad_Thetas[:,i,:] = torch.sum(vect_grad_Thetas, dim=1)





    def _compute_gamma(self, dnorm2, m):
        """
        dnorm2 : (B, M, P)  distances au carré ||X_{b,i} - Y_{b,j}||^2
        m      : nombre de points (M)

        Objectif : retourner gamma tel que
            k(x, y) = exp( - gamma * ||x - y||^2 )

        - Si sigma est fixé :
              gamma = 1 / (2 * sigma^2)
        - Si sigma est None :
              on estime sigma via la "median heuristic" :

                  h ≈ median( dnorm2 ) / (2 * log(m + 1))
                  sigma = sqrt(h)
                  gamma = 1 / (2 * sigma^2)
        """
        if self.sigma is None:
            # Median heuristic : on prend la médiane de toutes les distances au carré.
            # detach() pour ne pas backpropager à travers ce choix de sigma.
            median = torch.median(dnorm2.detach().flatten())

            # h ~ variance effective, divisée par 2 log(m+1) (stabilisation).
            h = median / (2.0 * math.log(m + 1.0))

            # sigma = sqrt(h), avec clamp pour éviter sigma=0.
            sigma_val = torch.sqrt(torch.clamp(h, min=1e-8))
        else:
            # sigma fourni par l'utilisateur (float ou tensor)
            sigma_val = torch.tensor(
                self.sigma,
                device=dnorm2.device,
                dtype=dnorm2.dtype
            )

        # gamma = 1 / (2 * sigma^2), avec petit epsilon pour la stabilité num.
        gamma = 1.0 / (1e-8 + 2.0 * sigma_val ** 2)

        # S'assurer que gamma est bien un tensor (scalaire) sur le bon device/dtype.
        if not torch.is_tensor(gamma):
            gamma = torch.tensor(gamma, device=dnorm2.device, dtype=dnorm2.dtype)

        return gamma