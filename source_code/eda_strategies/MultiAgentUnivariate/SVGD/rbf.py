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

    def forward(self, X, Y):
        """
        X : (B, M, N)
        Y : (B, P, N)

        Retourne :
            K : (B, M, P) avec K[b, i, j] = exp( - gamma * || X_{b,i} - Y_{b,j} ||^2 )
        """
        X = X.requires_grad_(True)
        Yd = Y.detach()
        # Produit scalaire par batch :
        # XX[b, i, j] = <X_{b,i}, X_{b,j}>
        XX = torch.matmul(X, X.transpose(-1, -2))        # (B, M, M)
        # XY[b, i, j] = <X_{b,i}, Y_{b,j}>
        XY = torch.matmul(X, Y.transpose(-1, -2))        # (B, M, M)
        # YY[b, i, j] = <Y_{b,i}, Y_{b,j}>
        YY = torch.matmul(Y, Y.transpose(-1, -2))        # (B, M, M)

        # Normes au carré :
        # diag_x[b, i] = ||X_{b,i}||^2
        diag_x = XX.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)  # (B, M, 1)
        # diag_y[b, j] = ||Y_{b,j}||^2
        diag_y = YY.diagonal(dim1=-2, dim2=-1).unsqueeze(-2)  # (B, 1, M)

        # Distances au carré :
        # dnorm2[b, i, j] = ||X_{b,i} - Y_{b,j}||^2
        #                 = ||X_{b,i}||^2 + ||Y_{b,j}||^2 - 2 <X_{b,i}, Y_{b,j}>
        dnorm2 = -2.0 * XY + diag_x + diag_y             # (B, M, M)

        # Calcule gamma à partir de dnorm2 (et éventuellement sigma)
        gamma = self._compute_gamma(dnorm2, m=X.size(-2))

        # Kernel RBF :
        # K[b, i, j] = exp( - gamma * dnorm2[b, i, j] )
        # gamma est un scalaire tensor -> broadcast sur (B, M, M)
        K = torch.exp(-gamma * dnorm2)
        gradX, = torch.autograd.grad(K.sum(), X, create_graph=True)
        grad_term = -gradX
        return K, grad_term 

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