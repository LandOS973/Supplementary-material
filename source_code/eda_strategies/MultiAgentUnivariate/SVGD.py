import torch
import torch.autograd as autograd

class SVGD:
    def __init__(self, kernel):
        self.kernel = kernel  # ex: RBF()

    def phi(self, theta, score):
        """
        theta : (M, N)  paramètres des M agents pour UNE instance
        score : (M, N)  direction 'score' pour chaque agent (ici g_RL déjà calculé)

        Retourne phi(theta) de taille (M, N).
        """
        # on veut des gradients par rapport à theta dans le calcul du noyau
        theta = theta.detach().requires_grad_(True)

        # noyau entre agents
        K = self.kernel(theta, theta)        # (M, M)

        # ∇_{theta_j} k(theta_j, theta_i)
        grad_K = -autograd.grad(K.sum(), theta)[0]  # (M, N)

        # direction SVGD : 1/M Σ_j [ k_ji * score_j + ∇_{theta_j} k_ji ]
        phi = (K.detach() @ score + grad_K) / theta.size(0)

        return phi.detach()
