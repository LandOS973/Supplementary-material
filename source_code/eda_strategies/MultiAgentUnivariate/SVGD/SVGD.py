import torch

class SVGD:
    def __init__(self, kernel, gamma=10.0, no_repulsion=False):
        self.kernel = kernel
        self.gamma = float(gamma)
        self.no_repulsion = bool(no_repulsion)

    def phi(self, thetas, score, probs=None):
        """
        B => Nombre d'instances
        N => Nombre de variables
        M => Nombre de particules (agents)
        theta : (B, M, N)   particles per batch (B instances, M agents, N dims)
        score : (B, M, N)   ∇_theta log p(theta) supplied by RL agents

        Standard SVGD update:
            φ_i = (1/M) * [ Σ_j k(θ_j, θ_i) * score_j  +  Σ_j ∇_θ_j k(θ_j, θ_i) ]
        """
        M = thetas.shape[1]
        if M == 1:
            return score / self.gamma
        K, grad_term = self.kernel(thetas, probs=probs)
        K = torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
        grad_term = torch.nan_to_num(grad_term, nan=0.0, posinf=0.0, neginf=0.0)
        if score.dim() == 4:
            score_term = (K.unsqueeze(-1).unsqueeze(-1) * score.unsqueeze(1)).sum(dim=2)
        else:
            score_term = torch.matmul(K, score)
        attraction = (score_term / self.gamma) / M
        repulsion = torch.zeros_like(attraction) if self.no_repulsion else grad_term / M
        phi = torch.nan_to_num(attraction + repulsion, nan=0.0, posinf=0.0, neginf=0.0)
        return phi
